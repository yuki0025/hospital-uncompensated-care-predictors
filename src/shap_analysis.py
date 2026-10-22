"""Step 7. SHAP interpretation of the XGBoost models (set C), computed out of fold: each hospital's SHAP values come from
the model that did not see it. Global importance, ranking stability across the outer fits, and domain-level importance
(SHAP values summed within feature domains)."""
import numpy as np
import pandas as pd

from src import features as F
from src import plotting as P
from src.io_utils import log, out_path, save_df, save_json

META = ['task', 'repeat', 'fold', 'row', 'base_value']


def load(spec):
    return pd.read_parquet(out_path(f'data/shap_{spec}.parquet'))


def importance_tables(sh, task, top_k=10):
    s = sh[sh.task == task]
    feats = [c for c in s.columns if c not in META]
    per_fit = s.groupby(['repeat', 'fold'])[feats].apply(lambda g: g.abs().mean())
    ranks = per_fit.rank(axis=1, ascending=False)
    glob = (s[s.repeat == 0][feats].abs().mean().sort_values(ascending=False))
    t = pd.DataFrame({'feature': glob.index, 'label': [F.LABELS.get(f, f) for f in glob.index], 'domain': [F.DOMAIN.get(f, '') for f in glob.index],
                      'mean_abs_shap': glob.values,
                      'median_rank': ranks.median()[glob.index].values,
                      'rank_iqr_lo': ranks.quantile(0.25)[glob.index].values, 'rank_iqr_hi': ranks.quantile(0.75)[glob.index].values,
                      'share_fits_top10': (ranks <= top_k).mean()[glob.index].values})
    # Kendall's W (concordance of rankings across fits)
    m, n = ranks.shape
    R = ranks.sum(axis=0)
    W = 12 * ((R - R.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))
    # domain-level importance: sum SHAP values within each domain per hospital, then mean absolute value per fit
    dom = {}
    for f in feats:
        dom.setdefault(F.DOMAIN.get(f, 'Other'), []).append(f)
    dom_fit = []
    for (rep, k), g in s.groupby(['repeat', 'fold']):
        dom_fit.append({dname: g[cols].sum(axis=1).abs().mean() for dname, cols in dom.items()})
    dfit = pd.DataFrame(dom_fit)
    dt = pd.DataFrame({'domain': dfit.columns, 'mean_abs_shap': dfit.mean().values, 'q25': dfit.quantile(0.25).values,
                       'q75': dfit.quantile(0.75).values}).sort_values('mean_abs_shap', ascending=False)
    return t, dt, float(W)


def run(d, S):
    sh = load('pooled_C')
    X = S['C'].reset_index(drop=True)
    out = {}
    for task, unit in [('clf', 'log-odds'), ('reg', 'UCR')]:
        t, dt, W = importance_tables(sh, task)
        save_df(t, f'tables/shap_{task}_feature_importance.csv')
        save_df(dt, f'tables/shap_{task}_domain_importance.csv')
        out[task] = {'kendall_w': W, 'top5': t.head(5)['label'].tolist(), 'top_domain': dt.iloc[0]['domain']}
        s0 = sh[(sh.task == task) & (sh.repeat == 0)].sort_values('row')
        stable = t[t.share_fits_top10 >= 0.8]['feature'].tolist()[:15] or t['feature'].tolist()[:15]
        P.beeswarm(s0[stable].values, X.loc[s0['row'].values, stable].values, [F.LABELS.get(f, f) for f in stable],
                   f'SHAP value ({unit} scale)', out_path(f'figures/shap_beeswarm_{task}.png'))
        log(f'   SHAP {task}: W={W:.2f}; top features {out[task]["top5"]}; top domain {out[task]["top_domain"]}')
    da = pd.read_csv(out_path('tables/shap_clf_domain_importance.csv'))
    dr = pd.read_csv(out_path('tables/shap_reg_domain_importance.csv'))
    P.domain_bars(da, dr, out_path('figures/shap_domain_importance.png'))
    # Main-text figure: domain importance (classification) next to the beeswarm of the classifier
    t_clf = pd.read_csv(out_path('tables/shap_clf_feature_importance.csv'))
    stable = t_clf[t_clf.share_fits_top10 >= 0.8]['feature'].tolist()[:15] or t_clf['feature'].tolist()[:15]
    s0 = sh[(sh.task == 'clf') & (sh.repeat == 0)].sort_values('row')
    P.shap_main_figure(da, s0[stable].values, X.loc[s0['row'].values, stable].values, [F.LABELS.get(f, f) for f in stable],
                       out_path('figures/figure4_shap.png'))
    # Subgroup models (Multimedia Appendix)
    sub = {}
    for spec, lab in [('cah_C', 'CAH'), ('noncah_C', 'Non-CAH')]:
        try:
            s = load(spec)
        except FileNotFoundError:
            continue
        for task in ['clf', 'reg']:
            t, dt, W = importance_tables(s, task)
            save_df(t, f'tables/shap_{spec}_{task}_feature_importance.csv')
            save_df(dt, f'tables/shap_{spec}_{task}_domain_importance.csv')
            sub[(lab, task)] = dt
            out[f'{spec}_{task}'] = {'kendall_w': W, 'top5': t.head(5)['label'].tolist(), 'top_domain': dt.iloc[0]['domain']}
    if sub:
        P.subgroup_domain_bars(sub, out_path('figures/shap_subgroup_domain_importance.png'))
    comparison_figures(d, S, sh)
    save_json(out, 'results/shap_summary.json')
    return out


def comparison_figures(d, S, sh_pooled, top_k=10):
    """SHAP comparison of the models for all hospitals, CAHs, and non-CAHs (each model trained in its own population; out-of-fold
    values of the first repeat), with the feature ranks of the 3 models side by side."""
    cah = d['cah'].values == 1
    pops = [('All hospitals', 'pooled_C', S['C'].reset_index(drop=True), sh_pooled),
            ('CAHs', 'cah_C', S['C'][cah].drop(columns=['cah']).reset_index(drop=True), None),
            ('Non-CAHs', 'noncah_C', S['C'][~cah].drop(columns=['cah']).reset_index(drop=True), None)]
    colors = [P.BLUE, P.ORANGE, P.AQUA]
    for task, unit, dunit, fig_name in [('clf', 'SHAP value (log-odds)', 'Mean |SHAP| (log-odds of UCR > 0.03)', 'figure6_shap_classification.png'),
                                        ('reg', 'SHAP value (UCR)', 'Mean |SHAP| (UCR scale)', 'figure5_shap_regression.png')]:
        panels, domains, ranks = [], [], []
        for k, (name, spec, X, sh) in enumerate(pops):
            sh = sh if sh is not None else load(spec)
            t, dt, _ = importance_tables(sh, task)
            s0 = sh[(sh.task == task) & (sh.repeat == 0)].sort_values('row')
            feats = t['feature'].tolist()[:top_k]
            panels.append((f'{"ABC"[k]}. {name}', s0[feats].values, X.loc[s0['row'].values, feats].values, [F.LABELS.get(f, f) for f in feats]))
            domains.append((name, dt, colors[k]))
            ranks.append(t.assign(rank=range(1, len(t) + 1)).set_index('feature')[['rank', 'mean_abs_shap']].add_suffix(f'_{spec}'))
        P.shap_comparison_figure(panels, domains, out_path(f'figures/{fig_name}'), unit, dunit)
        rk = pd.concat(ranks, axis=1)
        rk.insert(0, 'label', [F.LABELS.get(f, f) for f in rk.index])
        rk.insert(1, 'domain', [F.DOMAIN.get(f, '') for f in rk.index])
        save_df(rk.sort_values('rank_pooled_C').reset_index().rename(columns={'index': 'feature'}), f'tables/shap_{task}_rank_comparison.csv')
