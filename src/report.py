"""Step 10. Manuscript tables and figures, and a summary of key numbers (results/key_numbers.json).

Formatting follows the journal's rules: n/N (%) with one decimal place for denominators >= 100, P values as P<.001 or
P=.xx, and table footnotes as superscript letters."""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from src import parameters as config
from src import features as F
from src import metrics as M
from src import plotting as P
from src.io_utils import log, out_path, save_df, save_json


# ---------------------------------------------------------------------------------------------- formatting helpers
def pct(n, N):
    return f'{n:,}/{N:,} ({100 * n / N:.0f}%)' if N < 100 else f'{n:,}/{N:,} ({100 * n / N:.1f}%)'


def fp(p):
    if p is None or not np.isfinite(p):
        return ''
    if p < .001:
        return '<.001'
    return f'{p:.3f}'.lstrip('0') if p < .01 else f'{p:.2f}'.lstrip('0')


def ci(est, lo, hi, d=2):
    return f'{est:.{d}f} ({lo:.{d}f}-{hi:.{d}f})'


def med_iqr(v, d=3, scale=1.0):
    v = pd.Series(v).dropna() * scale
    return f'{v.median():.{d}f} ({v.quantile(.25):.{d}f}-{v.quantile(.75):.{d}f})'


# ---------------------------------------------------------------------------------------------- Table 1
def table1(d):
    groups = [('Overall', np.ones(len(d), bool)), ('CAH', d.cah.values == 1), ('Non-CAH', d.cah.values == 0)]
    rows = []

    def cat(name, series, levels):
        tab = pd.crosstab(series, d.cah)
        p = stats.chi2_contingency(tab)[1] if tab.shape[0] > 1 else np.nan
        rows.append({'Characteristic': name, **{g: '' for g, _ in groups}, 'P value': fp(p)})
        for lv, lab in levels:
            rows.append({'Characteristic': f'  {lab}', **{g: pct(int((series[s] == lv).sum()), int(s.sum())) for g, s in groups}, 'P value': ''})

    def cont(name, v, d_=3, scale=1.0):
        p = stats.mannwhitneyu(v[d.cah == 1].dropna(), v[d.cah == 0].dropna()).pvalue
        rows.append({'Characteristic': name, **{g: med_iqr(v[s], d_, scale) for g, s in groups}, 'P value': fp(p)})

    cat('Ownership', d['own'], [('Not-for-profit', 'Not-for-profit'), ('For-profit', 'For-profit'), ('Government', 'Government (nonfederal)')])
    cat('Hospital size', d['size'], [('Small', 'Small (<100 beds)'), ('Medium', 'Medium (100-399 beds)'), ('Large', 'Large (>=400 beds)')])
    cat('Location', d['rural'], [(1, 'Rural'), (0, 'Urban')])
    cat('Teaching status', d['teach'], [('Non Teaching', 'Nonteaching'), ('Minor', 'Minor teaching'), ('Major', 'Major teaching')])
    cat('System membership', d['system'], [(1, 'System member')])
    cat('Census region', d['region'], [('Northeast', 'Northeast'), ('Midwest', 'Midwest'), ('South', 'South'), ('West', 'West')])
    cat('Medicaid expansion state (as of January 1, 2023)', d['expansion'], [(1, 'Yes')])
    cat('Responded to the AHA Annual Survey', d['respondent'], [(1, 'Yes')])
    cont('County uninsured rate, age <65 years (%), median (IQR)', d['sahie2023_pctui_u65'], 1)
    cont('County poverty rate (%), median (IQR)', d['saipe2023_pov_pct_all'], 1)
    cont('Uncompensated care ratio, median (IQR)', d['ucr'], 4)
    cat('High uncompensated care burden (UCR >0.03)', d['high_ucr'], [(1, 'Yes')])
    cont('Charity care and bad debt cost, % of operating expenses, median (IQR)', d['ucr30_opex'], 1, 100)
    t = pd.DataFrame(rows)
    t.columns = ['Characteristic', f'Overall (n={len(d):,})', f'CAH (n={int(d.cah.sum()):,})', f'Non-CAH (n={int((d.cah == 0).sum()):,})', 'P value']
    save_df(t, 'manuscript_tables/table1_characteristics.csv')
    return t


# ---------------------------------------------------------------------------------------------- Table 2
def table2():
    m1 = pd.read_csv(out_path('tables/beta_M1_adjusted.csv'))
    m3 = pd.read_csv(out_path('tables/beta_M3_adjusted.csv'))
    rows = []
    for var, g in m1.groupby('variable', sort=False):
        if var == 'CAH x Medicaid expansion':
            continue
        rows.append({'Characteristic': var, 'Adjusted mean UCR (95% CI)': '', 'Ratio (95% CI), core model': '', 'P value': '',
                     'Ratio (95% CI), extended model': ''})
        for _, r in g.iterrows():
            r3 = m3[(m3.variable == var) & (m3.level == r.level)]
            ref = pd.isna(r.get('ratio'))
            rows.append({'Characteristic': f'  {r.level}', 'Adjusted mean UCR (95% CI)': ci(r.adj_mean, r.adj_mean_lo, r.adj_mean_hi, 4),
                         'Ratio (95% CI), core model': 'Reference' if ref else ci(r.ratio, r.ratio_lo, r.ratio_hi),
                         'P value': '' if ref else fp(r.p),
                         'Ratio (95% CI), extended model': 'Reference' if ref or r3.empty else ci(r3.iloc[0].ratio, r3.iloc[0].ratio_lo, r3.iloc[0].ratio_hi)})
    it = m1[m1.variable == 'CAH x Medicaid expansion']
    rows.append({'Characteristic': 'CAH x Medicaid expansion', 'Adjusted mean UCR (95% CI)': '', 'Ratio (95% CI), core model': '', 'P value': '',
                 'Ratio (95% CI), extended model': ''})
    for _, r in it.iterrows():
        if pd.notna(r.get('adj_mean')):
            rows.append({'Characteristic': f'  {r.level}', 'Adjusted mean UCR (95% CI)': ci(r.adj_mean, r.adj_mean_lo, r.adj_mean_hi, 4),
                         'Ratio (95% CI), core model': '', 'P value': '', 'Ratio (95% CI), extended model': ''})
        else:
            rows.append({'Characteristic': f'  {r.level}', 'Adjusted mean UCR (95% CI)': f'{r.ame:+.4f} ({r.ame_lo:+.4f} to {r.ame_hi:+.4f})',
                         'Ratio (95% CI), core model': '', 'P value': fp(r.p), 'Ratio (95% CI), extended model': ''})
    t = pd.DataFrame(rows)
    save_df(t, 'manuscript_tables/table2_beta_regression.csv')
    return t


# ---------------------------------------------------------------------------------------------- performance tables
LEARNER_LABELS = {'lr_cah': 'Logistic regression, CAH status only', 'lr5': 'Logistic regression, 5 structural variables',
                  'lr5_exp': 'Logistic regression, 5 variables + Medicaid expansion', 'lr_a': 'Logistic regression, core set A',
                  'beta_a': 'Beta regression, core set A', 'null': 'Training-sample mean', 'en': 'Elastic net', 'rf': 'Random forest',
                  'mlp': 'Multilayer perceptron', 'xgb': 'XGBoost'}


def perf_summary():
    specs = ['pooled_bench', 'pooled_A', 'pooled_B', 'pooled_C', 'pooled_C_noCAH', 'cah_C', 'noncah_C', 'secondary_C'] + \
            [f'cutoff_{c}' for c in [0.02, 0.025, 0.035, 0.04]]
    from src import ml
    fs = [pd.read_csv(out_path(f'tables/ml_folds_{s}.csv')) for s in specs if out_path(f'tables/ml_folds_{s}.csv').exists()]
    allf = pd.concat(fs)
    sm = ml.summarize(allf)
    save_df(sm, 'tables/ml_performance_summary.csv')
    return allf, sm


def table3(sm, folds):
    rows = []
    order = [('pooled_bench', 'lr_cah'), ('pooled_bench', 'lr5'), ('pooled_bench', 'lr5_exp'), ('pooled_bench', 'lr_a'), ('pooled_bench', 'beta_a'),
             ('pooled_A', 'xgb'), ('pooled_B', 'xgb'), ('pooled_C', 'en'), ('pooled_C', 'mlp'), ('pooled_C', 'rf'), ('pooled_C', 'xgb'),
             ('pooled_C_noCAH', 'xgb')]
    setname = {'pooled_A': 'set A', 'pooled_B': 'set B', 'pooled_C': 'set C', 'pooled_C_noCAH': 'set C without CAH indicator'}
    ref_c = folds[(folds.spec == 'pooled_C') & (folds.learner == 'xgb') & (folds.task == 'clf')]
    for spec, ln in order:
        c = sm[(sm.spec == spec) & (sm.learner == ln) & (sm.task == 'clf')]
        r = sm[(sm.spec == spec) & (sm.learner == ln) & (sm.task == 'reg')]
        lab = LEARNER_LABELS[ln] + (f', {setname[spec]}' if spec in setname else '')
        row = {'Model': lab}
        if not c.empty:
            c = c.iloc[0]
            row.update({'AUROC (95% CI)': ci(c.auroc, c.auroc_lo, c.auroc_hi), 'AUPRC': f'{c.auprc:.2f}', 'F1 score, high-burden class': f'{c.f1:.2f}',
                        'Brier score': f'{c.brier:.3f}', 'Calibration slope': f'{c.cal_slope:.2f}'})
        if not r.empty:
            r = r.iloc[0]
            row.update({'R2 (95% CI)': ci(r.r2, r.r2_lo, r.r2_hi), 'RMSE': f'{r.rmse:.4f}', 'MAE': f'{r.mae:.4f}'})
        rows.append(row)
    t = pd.DataFrame(rows).fillna('NA')
    save_df(t, 'manuscript_tables/table3_model_performance.csv')
    return t


def paired_comparisons(folds):
    """Paired differences between models on the same outer folds (corrected resampled t test)."""
    from src import ml
    setname = {'pooled_A': 'set A', 'pooled_B': 'set B', 'pooled_C': 'set C', 'pooled_C_noCAH': 'set C without CAH indicator'}
    lab = lambda s, ln: LEARNER_LABELS[ln] + (f', {setname[s]}' if s in setname else '')
    pick = lambda s, ln, task: folds[(folds.spec == s) & (folds.learner == ln) & (folds.task == task)]
    xc, rc, xb, xa = ('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_B', 'xgb'), ('pooled_A', 'xgb')
    pairs = [('clf', 'auroc', xc, ('pooled_bench', 'lr_cah')), ('clf', 'auroc', xc, ('pooled_bench', 'lr5')),
             ('clf', 'auroc', xc, ('pooled_bench', 'lr5_exp')), ('clf', 'auroc', xc, ('pooled_bench', 'lr_a')),
             ('clf', 'auroc', xc, rc), ('clf', 'auroc', xc, xb), ('clf', 'auroc', xb, xa), ('clf', 'auroc', xa, ('pooled_bench', 'lr_a')),
             ('clf', 'auroc', xc, ('pooled_C_noCAH', 'xgb')),
             ('reg', 'r2', xc, ('pooled_bench', 'beta_a')), ('reg', 'r2', xc, rc), ('reg', 'r2', xc, xb), ('reg', 'r2', xb, xa),
             ('reg', 'r2', xa, ('pooled_bench', 'beta_a'))]
    rows = []
    for task, metric, a, b in pairs:
        fa, fb = pick(*a, task), pick(*b, task)
        if fa.empty or fb.empty:
            continue
        diff, p = ml.paired_comparison(fa, fb, metric)
        rows.append({'task': task, 'metric': metric, 'model_a': lab(*a), 'model_b': lab(*b), 'mean_a': fa[metric].mean(),
                     'mean_b': fb[metric].mean(), 'difference_a_minus_b': diff, 'p_corrected_t': p})
    t = pd.DataFrame(rows)
    save_df(t, 'tables/ml_paired_comparisons.csv')
    return t


def appendix_performance(sm):
    keep = ['spec', 'task', 'learner', 'n_fits', 'auroc', 'auroc_lo', 'auroc_hi', 'auprc', 'brier', 'scaled_brier', 'cal_intercept', 'cal_slope',
            'threshold', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'tp_sum_repeat0', 'fp_sum_repeat0', 'fn_sum_repeat0', 'tn_sum_repeat0', 'r2', 'r2_lo', 'r2_hi', 'rmse', 'mae', 'spearman']
    t = sm[[c for c in keep if c in sm.columns]].copy()
    t.insert(3, 'model', t['learner'].map(LEARNER_LABELS))
    save_df(t, 'manuscript_tables/appendix_ml_performance_all.csv')


# ---------------------------------------------------------------------------------------------- figures
def figures(d):
    P.workflow_diagram(out_path('figures/figure1_workflow.png'))
    flow = pd.read_csv(out_path('tables/flow.csv'))
    side = {'No FY2023 Medicare cost report': ('no FY2023 Medicare cost report', ''),
            'US territory': ('US territory', ''),
            'Not a general medical and surgical hospital': ('not a general medical and surgical hospital', 'psychiatric, rehabilitation, long-term acute care, and other specialty hospitals'),
            'No uncompensated care or patient revenue reported': ('no uncompensated care or patient revenue reported', ''),
            f'Implausible ratio (>= {config.UCR_MAX})': (f'uncompensated care ratio >= {config.UCR_MAX}', '')}
    main = ['AHA Annual Survey 2023 hospitals', 'Linked to an FY2023 Medicare cost report', 'Located in the 50 states or DC',
            'General medical and surgical hospitals', 'Reported uncompensated care and patient revenue', 'Analytic sample']
    f = flow[flow.step != 'Analytic sample'].reset_index(drop=True)
    recs = []
    for i, r in f.iterrows():
        rec = {'main_label': main[i], 'n_remaining': int(r.n_remaining), 'n_excluded': r.n_excluded}
        if r.step in side:
            rec['side_label'], rec['side_detail'] = side[r.step]
            det = str(r.detail) if isinstance(r.detail, str) else ''
            if r.step == 'No uncompensated care or patient revenue reported' and det:
                import re as _re
                z = _re.search(r'zero (\d+) \(federal (\d+), Maryland (\d+)\)', det)
                zr = _re.search(r'zero patient revenue (\d+)', det)
                rec['side_detail'] = (f'zero uncompensated care, {z.group(1)} (federal, {z.group(2)}; Maryland, {z.group(3)}); '
                                      f'zero patient revenue, {zr.group(1)}; categories overlap') if z and zr else det
        recs.append(rec)
    recs[-1]['main_label'] = 'Analytic sample'
    P.flow_diagram(pd.DataFrame(recs), [('Critical access hospitals', int(d.cah.sum())), ('Non-critical access hospitals', int((d.cah == 0).sum()))],
                   out_path('figures/figure2_flow.png'))
    m1 = pd.read_csv(out_path('tables/beta_M1_adjusted.csv'))
    cells = []
    for lvl, (c, e) in {'Non-CAH, non-expansion': ('Non-CAH', 'Non-expansion'), 'Non-CAH, expansion': ('Non-CAH', 'Expansion'),
                        'CAH, non-expansion': ('CAH', 'Non-expansion'), 'CAH, expansion': ('CAH', 'Expansion')}.items():
        r = m1[(m1.variable == 'CAH x Medicaid expansion') & (m1.level == lvl)].iloc[0]
        cells.append({'cah': c, 'expansion': e, 'mean': r.adj_mean, 'lo': r.adj_mean_lo, 'hi': r.adj_mean_hi})
    P.cah_expansion_plot(cells, out_path('figures/figure3_cah_expansion.png'))
    # ROC and calibration curves (pooled, first repeat, out-of-fold)
    oof = pd.concat([pd.read_parquet(out_path(f'data/oof_{s}.parquet')) for s in ['pooled_bench', 'pooled_C']])
    oof = oof[(oof.task == 'clf') & (oof.repeat == 0)]
    curves, cal = [], []
    for spec, ln in [('pooled_C', 'xgb'), ('pooled_C', 'rf'), ('pooled_C', 'mlp'), ('pooled_C', 'en'), ('pooled_bench', 'lr5_exp'), ('pooled_bench', 'lr_cah')]:
        o = oof[(oof.spec == spec) & (oof.learner == ln)]
        fpr, tpr, _ = roc_curve(o.y, o.pred)
        curves.append((LEARNER_LABELS[ln], fpr, tpr, roc_auc_score(o.y, o.pred)))
        if ln in ('xgb', 'rf', 'lr5_exp'):
            yb, pb = calibration_curve(o.y, o.pred, n_bins=10, strategy='quantile')
            cal.append((LEARNER_LABELS[ln], pb, yb))
    P.roc_panel(curves, out_path('figures/appendix_roc_pooled.png'))
    P.calibration_panel(cal, out_path('figures/appendix_calibration_pooled.png'))
    cut = pd.read_csv(out_path('tables/ml_performance_summary.csv'))
    rows = []
    for c in [0.02, 0.025, 0.03, 0.035, 0.04]:
        spec = 'pooled_C' if c == 0.03 else f'cutoff_{c}'
        bspec = 'pooled_bench' if c == 0.03 else f'cutoff_{c}'
        for ln, spc in [('xgb', spec), ('lr5_exp', bspec)]:
            r = cut[(cut.spec == spc) & (cut.learner == ln) & (cut.task == 'clf')]
            if not r.empty:
                rows.append({'cutoff': c, 'model': LEARNER_LABELS[ln], 'auroc': r.iloc[0].auroc, 'auprc': r.iloc[0].auprc,
                             'prevalence': r.iloc[0].prevalence, 'n_high': int((d['ucr'] > c).sum()), 'n': len(d)})
    cdf = pd.DataFrame(rows)
    save_df(cdf, 'manuscript_tables/appendix_cutoff_robustness.csv')
    if not cdf.empty:
        P.line_by_cutoff(cdf, out_path('figures/appendix_cutoff_robustness.png'))


def key_numbers(d, sm):
    k = {}
    fl = pd.read_csv(out_path('tables/flow.csv'))
    k['flow'] = fl[['step', 'n_remaining', 'n_excluded', 'detail']].to_dict('records')
    k['n'] = len(d); k['cah'] = int(d.cah.sum()); k['noncah'] = int((d.cah == 0).sum())
    k['high_ucr'] = pct(int(d.high_ucr.sum()), len(d)); k['high_ucr_cah'] = pct(int(d.loc[d.cah == 1, 'high_ucr'].sum()), int(d.cah.sum()))
    k['high_ucr_noncah'] = pct(int(d.loc[d.cah == 0, 'high_ucr'].sum()), int((d.cah == 0).sum()))
    k['cutoff_percentile'] = float((d.ucr <= config.UCR_CUTOFF).mean())
    k['median_ucr'] = float(d.ucr.median())
    nr = d['Medicare Provider ID'].notna()
    k['net_to_gross_median'] = float((d['Net patient revenues'] / d['Total patient revenues']).median())
    k['county_uninsured_median_by_expansion'] = {str(int(e)): float(g['sahie2023_pctui_u65'].median()) for e, g in d.groupby('expansion')}
    k['expansion_by_region'] = {r: {'expansion': int(((d.region == r) & (d.expansion == 1)).sum()), 'nonexpansion': int(((d.region == r) & (d.expansion == 0)).sum())}
                                for r in ['Northeast', 'Midwest', 'South', 'West']}
    ok = d['ucr30_opex'].notna()
    k['n_outcome_at_cost'] = {'all': int(ok.sum()), 'cah': int((ok & (d.cah == 1)).sum()), 'noncah': int((ok & (d.cah == 0)).sum())}
    for f in ['inferential_summary', 'missing_mechanism', 'temporal_validation']:
        p = out_path(f'results/{f}.json')
        if p.exists():
            k[f] = json.loads(p.read_text())
    save_json(k, 'results/key_numbers.json')
    return k


def run():
    d = pd.read_parquet(out_path('data/analytic.parquet'))
    folds, sm = perf_summary()
    table1(d)
    table2()
    table3(sm, folds)
    appendix_performance(sm)
    paired_comparisons(folds)
    figures(d)
    key_numbers(d, sm)
    log('   report: tables and figures written')
