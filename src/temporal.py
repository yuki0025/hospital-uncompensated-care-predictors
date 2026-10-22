"""Step 9. Temporal validation: do the models identify the hospitals that have a high uncompensated care ratio in the
following fiscal year (FY2024)?

FY2024 outcomes are computed from the public CMS HCRIS release with the same definitions (Worksheet S-10 Part I line 31 /
Worksheet G-3 line 1). Each hospital's prediction is its out-of-fold prediction from the repeated cross-validation, i.e.
from models trained on FY2023 data of other hospitals only, averaged over the outer repeats; predictors are 2023
characteristics. The hospital's own FY2023 ratio is reported as a reference (it requires the prior-year cost report)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src import parameters as config
from src import hcris
from src import metrics as M
from src import plotting as P
from src.io_utils import log, out_path, save_df, save_json

LABELS = {'xgb': 'XGBoost, set C', 'rf': 'Random forest, set C', 'mlp': 'Multilayer perceptron, set C', 'en': 'Elastic net, set C',
          'lr5_exp': 'Logistic regression, 5 variables + Medicaid expansion', 'lr_a': 'Logistic regression, core set A',
          'beta_a': 'Beta regression, core set A'}


def _boot_ci(y, p, fn, n=1000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _oof_avg(spec, task, learner):
    o = pd.read_parquet(out_path(f'data/oof_{spec}.parquet'))
    o = o[(o.task == task) & (o.learner == learner)]
    return o.groupby('row')['pred'].mean()


def run(d, S):
    ccn = d['Medicare Provider ID_cms'].astype(str).str.zfill(6).values
    h23 = hcris.extract(config.HCRIS_DIR, 2023).set_index('prvdr_num')
    h24 = hcris.extract(config.HCRIS_DIR, 2024).set_index('prvdr_num')
    # data-vintage check: the FY2023 extract used in the study vs the current public HCRIS release
    v = pd.DataFrame({'ucr': d['ucr'].values, 'ccn': ccn}).join(h23[['total_uc_cost_l31', 'total_patient_revenues']], on='ccn')
    v['ucr_h'] = v['total_uc_cost_l31'] / v['total_patient_revenues'].replace(0, np.nan)
    vv = v.dropna(subset=['ucr_h'])
    vintage = {'n_matched': int(len(vv)), 'spearman': float(stats.spearmanr(vv.ucr, vv.ucr_h).correlation),
               'same_high_low_class': float(((vv.ucr > config.UCR_CUTOFF) == (vv.ucr_h > config.UCR_CUTOFF)).mean())}
    t = pd.DataFrame({'row': np.arange(len(d)), 'ccn': ccn, 'ucr23': d['ucr'].values, 'high23': d['high_ucr'].values}).join(
        h24[['total_uc_cost_l31', 'total_patient_revenues']], on='ccn')
    t['ucr24'] = t['total_uc_cost_l31'] / t['total_patient_revenues'].replace(0, np.nan)
    ok = (t['total_uc_cost_l31'] > 0) & (t['total_patient_revenues'] > 0) & (t['ucr24'] < config.UCR_MAX)
    flow = {'analytic_sample': int(len(t)), 'no_fy2024_report': int(t['total_patient_revenues'].isna().sum()),
            'fy2024_no_uc_or_revenue_or_implausible': int((t['total_patient_revenues'].notna() & ~ok).sum()), 'validation_n': int(ok.sum())}
    t = t[ok].copy()
    t['high24'] = (t['ucr24'] > config.UCR_CUTOFF).astype(int)
    rows, curves = [], []
    folds = pd.concat([pd.read_csv(out_path(f'tables/ml_folds_{s}.csv')) for s in ['pooled_C', 'pooled_bench']])
    for task, items in [('clf', [('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_C', 'mlp'), ('pooled_C', 'en'),
                                 ('pooled_bench', 'lr5_exp'), ('pooled_bench', 'lr_a')]),
                        ('reg', [('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_C', 'mlp'), ('pooled_C', 'en'), ('pooled_bench', 'beta_a')])]:
        for spec, lname in items:
            pred = _oof_avg(spec, task, lname).reindex(t['row']).values
            for year, ytrue in [('FY2023', t['high23'].values if task == 'clf' else t['ucr23'].values),
                                ('FY2024', t['high24'].values if task == 'clf' else t['ucr24'].values)]:
                if task == 'clf':
                    tau = float(folds[(folds.spec == spec) & (folds.learner == lname) & (folds.task == 'clf')]['threshold'].median())
                    m = M.clf_metrics(ytrue, pred, tau, float(d['high_ucr'].mean()))
                    m['auroc_lo'], m['auroc_hi'] = _boot_ci(ytrue, pred, roc_auc_score)
                    m['auprc_lo'], m['auprc_hi'] = _boot_ci(ytrue, pred, average_precision_score)
                    if year == 'FY2024' and lname in ('xgb', 'lr5_exp'):
                        fpr, tpr, _ = roc_curve(ytrue, pred)
                        curves.append((LABELS[lname], fpr, tpr, m['auroc']))
                else:
                    m = M.reg_metrics(ytrue, pred)
                    m['r2_lo'], m['r2_hi'] = _boot_ci(ytrue, pred, lambda a, b: 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
                rows.append({'task': task, 'model': LABELS[lname], 'outcome_year': year, 'n': len(t), **m})
    ref = {'auroc_prior_year_ratio_for_fy2024_high': float(roc_auc_score(t['high24'], t['ucr23'])),
           'agreement_high_2023_2024': float((t['high23'] == t['high24']).mean()),
           'high23_high24': int(((t.high23 == 1) & (t.high24 == 1)).sum()), 'high23_low24': int(((t.high23 == 1) & (t.high24 == 0)).sum()),
           'low23_high24': int(((t.high23 == 0) & (t.high24 == 1)).sum()), 'low23_low24': int(((t.high23 == 0) & (t.high24 == 0)).sum()),
           'prevalence_2024': float(t['high24'].mean()), 'spearman_ucr_2023_2024': float(stats.spearmanr(t.ucr23, t.ucr24).correlation)}
    res = pd.DataFrame(rows)
    save_df(res, 'tables/temporal_validation.csv')
    roc_figure(t, res)
    save_json({'vintage_check_fy2023': vintage, 'flow': flow, 'reference': ref}, 'results/temporal_validation.json')
    if curves:
        P.roc_panel(curves, out_path('figures/temporal_validation_roc.png'), title='High burden in FY2024 (out-of-fold predictions)')
    log('   temporal validation:', flow, '| vintage', {k: round(x, 3) if isinstance(x, float) else x for k, x in vintage.items()})
    log('   temporal AUROC FY2024:', res[(res.task == 'clf') & (res.outcome_year == 'FY2024')][['model', 'auroc', 'auroc_lo', 'auroc_hi']].round(3).to_dict('records'))
    log('   reference:', {k: round(x, 3) if isinstance(x, float) else x for k, x in ref.items()})
    return res


def roc_figure(t, res):
    """Main-text ROC figure. (A) FY2023: out-of-fold predictions of the first repeat; the legend gives the mean AUROC over the 15 test
    folds (as in Table 3). (B) FY2024 outcomes: out-of-fold predictions averaged over the repeats (as in the temporal validation)."""
    sm = pd.read_csv(out_path('tables/ml_performance_summary.csv'))
    oof = pd.concat([pd.read_parquet(out_path(f'data/oof_{s}.parquet')) for s in ['pooled_C', 'pooled_bench']])
    oof = oof[(oof.task == 'clf') & (oof.repeat == 0)]
    style = {'xgb': (P.BLUE, '-'), 'rf': (P.AQUA, '-'), 'lr5_exp': (P.ORANGE, '--'), 'lr_cah': (P.NEUTRAL, ':')}
    names = {'xgb': 'XGBoost, set C', 'rf': 'Random forest, set C', 'lr5_exp': 'Logistic regression, 5 variables + expansion',
             'lr_cah': 'CAH status only'}
    pa = []
    for spec, ln in [('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_bench', 'lr5_exp'), ('pooled_bench', 'lr_cah')]:
        o = oof[(oof.spec == spec) & (oof.learner == ln)]
        fpr, tpr, _ = roc_curve(o.y, o.pred)
        auc = float(sm[(sm.spec == spec) & (sm.learner == ln) & (sm.task == 'clf')].auroc.iloc[0])
        pa.append((names[ln], fpr, tpr, auc, *style[ln]))
    pb = []
    for spec, ln in [('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_bench', 'lr5_exp')]:
        pred = _oof_avg(spec, 'clf', ln).reindex(t['row']).values
        fpr, tpr, _ = roc_curve(t['high24'].values, pred)
        auc = float(res[(res.task == 'clf') & (res.outcome_year == 'FY2024') & (res.model == LABELS[ln])].auroc.iloc[0])
        pb.append((names[ln], fpr, tpr, auc, *style[ln]))
    P.roc_two_panel([('A. Fiscal year 2023 (cross-validation)', pa), ('B. Fiscal year 2024 (temporal validation)', pb)],
                    out_path('figures/figure4_roc.png'))
