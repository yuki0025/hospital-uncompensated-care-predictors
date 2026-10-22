"""Step 6. How much of the discrimination reflects critical access status? (reviewer comment CO4)

Uses the out-of-fold predictions of the pooled models: benchmarks (CAH alone, five-variable logistic regression, plus
Medicaid expansion, core set), the set C learners, and set C without the CAH indicator."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import metrics as M
from src.io_utils import log, out_path, save_df


def _oof(spec):
    return pd.read_parquet(out_path(f'data/oof_{spec}.parquet'))


def run(d):
    cah = d['cah'].values
    oof = pd.concat([_oof('pooled_bench'), _oof('pooled_C'), _oof('pooled_C_noCAH')])
    oof = oof[oof.task == 'clf']
    folds = pd.concat([pd.read_csv(out_path(f'tables/ml_folds_{s}.csv')) for s in ['pooled_bench', 'pooled_C', 'pooled_C_noCAH']])
    folds = folds[folds.task == 'clf']
    models = [('pooled_bench', 'lr_cah', 'CAH status only (logistic)'), ('pooled_bench', 'lr5', 'Five-variable logistic regression'),
              ('pooled_bench', 'lr5_exp', 'Five variables + Medicaid expansion'), ('pooled_bench', 'lr_a', 'Core set A (logistic)'),
              ('pooled_C', 'en', 'Elastic net, set C'), ('pooled_C', 'mlp', 'MLP, set C'), ('pooled_C', 'rf', 'Random forest, set C'),
              ('pooled_C', 'xgb', 'XGBoost, set C'), ('pooled_C_noCAH', 'xgb', 'XGBoost, set C without CAH indicator')]
    ref = folds[(folds.spec == 'pooled_C') & (folds.learner == 'xgb')]
    rows = []
    for spec, lname, label in models:
        f = folds[(folds.spec == spec) & (folds.learner == lname)]
        if f.empty:
            continue
        mean, lo, hi = M.corrected_ci(f['auroc'], f['n_train'].mean(), f['n_test'].mean())
        row = {'model': label, 'auroc': mean, 'auroc_lo': lo, 'auroc_hi': hi}
        if not (spec == 'pooled_C' and lname == 'xgb'):
            j = ref.set_index(['repeat', 'fold'])['auroc'] - f.set_index(['repeat', 'fold'])['auroc']
            row['diff_vs_xgb_C'], row['p_corrected_t'] = M.corrected_ttest(j.dropna().values, f['n_train'].mean(), f['n_test'].mean())
            # DeLong test on pooled out-of-fold predictions, per repeat (median P reported)
            ps = []
            for rep in sorted(oof.repeat.unique()):
                a = oof[(oof.spec == 'pooled_C') & (oof.learner == 'xgb') & (oof.repeat == rep)].sort_values('row')
                b = oof[(oof.spec == spec) & (oof.learner == lname) & (oof.repeat == rep)].sort_values('row')
                ps.append(M.delong(a['y'].values, a['pred'].values, b['pred'].values)[2])
            row['p_delong_median'] = float(np.median(ps))
        # within-stratum AUROC of the pooled model
        for lab, flag in [('within_cah', 1), ('within_noncah', 0)]:
            vals = []
            for rep in sorted(oof.repeat.unique()):
                o = oof[(oof.spec == spec) & (oof.learner == lname) & (oof.repeat == rep)]
                sel = cah[o['row'].values] == flag
                if sel.sum() and o['y'].values[sel].min() != o['y'].values[sel].max():
                    vals.append(roc_auc_score(o['y'].values[sel], o['pred'].values[sel]))
            row[f'auroc_{lab}'] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    save_df(t, 'tables/co4_cah_question.csv')

    # pair decomposition of the pooled AUROC (XGBoost set C and five-variable logistic, repeat 0)
    prow = []
    for spec, lname, label in [('pooled_C', 'xgb', 'XGBoost, set C'), ('pooled_bench', 'lr5', 'Five-variable logistic regression'),
                               ('pooled_bench', 'lr_cah', 'CAH status only (logistic)')]:
        o = oof[(oof.spec == spec) & (oof.learner == lname) & (oof.repeat == 0)].sort_values('row')
        y, p, c = o['y'].values, o['pred'].values, cah[o['row'].values]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        total = len(pos) * len(neg)
        for pc, nc, name in [(1, 1, 'high CAH vs low CAH'), (0, 0, 'high non-CAH vs low non-CAH'),
                             (1, 0, 'high CAH vs low non-CAH'), (0, 1, 'high non-CAH vs low CAH')]:
            pp, nn = pos[c[pos] == pc], neg[c[neg] == nc]
            if len(pp) == 0 or len(nn) == 0:
                continue
            diff = p[pp][:, None] - p[nn][None, :]
            conc = (diff > 0).mean() + 0.5 * (diff == 0).mean()
            prow.append({'model': label, 'pair_type': name, 'share_of_pairs': len(pp) * len(nn) / total, 'concordance': conc})
    save_df(pd.DataFrame(prow), 'tables/co4_pair_decomposition.csv')
    log('   CO4 table:', t[['model', 'auroc', 'auroc_within_cah', 'auroc_within_noncah']].round(3).to_dict('records'))
    return t


def identifier_control(d):
    """Negative control for reviewer comment CO1: how well would identifiers alone predict high burden? (5-fold CV,
    XGBoost on the numeric identifier). Identifiers never enter the study models; this shows why they must not."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from src import parameters as config
    y = d['high_ucr'].values
    rows = []
    ids = {'AHA identifier': pd.to_numeric(d['AHA ID'].str.extract(r'(\d+)')[0], errors='coerce'),
           'Medicare provider number (CCN)': pd.to_numeric(d['Medicare Provider ID_cms'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')}
    for lab, v in ids.items():
        X = pd.DataFrame({'id': v.values})
        p = cross_val_predict(xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config.SEED, n_jobs=config.N_JOBS),
                              X, y, cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method='predict_proba')[:, 1]
        pc = cross_val_predict(xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config.SEED, n_jobs=config.N_JOBS),
                               X, d['cah'].values, cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), method='predict_proba')[:, 1]
        rows.append({'identifier': lab, 'auroc_high_burden': roc_auc_score(y, p), 'auroc_cah_status': roc_auc_score(d['cah'].values, pc)})
    t = pd.DataFrame(rows)
    save_df(t, 'tables/co1_identifier_negative_control.csv')
    log('   CO1 identifier-only control:', t.round(3).to_dict('records'))
    return t
