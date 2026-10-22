"""Step 4. Inferential analysis: multivariable beta regression (logit link) of the uncompensated care ratio with standard
errors clustered by state, adjusted (marginally standardized) means, and sensitivity analyses."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import parameters as config
from src import missing
from src.io_utils import log, save_df, save_json

warnings.filterwarnings('ignore')
expit = lambda z: 1 / (1 + np.exp(-z))

# ---------------------------------------------------------------------------------------------- design and fitting
SET_A_CONTRASTS = {
    'Critical access hospital': ('cah', {'No': {'cah': 0}, 'Yes': {'cah': 1}}),
    'Ownership': (None, {'Not-for-profit': {'own_forprofit': 0, 'own_government': 0}, 'For-profit': {'own_forprofit': 1, 'own_government': 0},
                         'Government': {'own_forprofit': 0, 'own_government': 1}}),
    'Hospital size': (None, {'Small': {'size_medium': 0, 'size_large': 0}, 'Medium': {'size_medium': 1, 'size_large': 0},
                             'Large': {'size_medium': 0, 'size_large': 1}}),
    'Location': ('rural', {'Urban': {'rural': 0}, 'Rural': {'rural': 1}}),
    'System member': ('system', {'No': {'system': 0}, 'Yes': {'system': 1}}),
    'Teaching status': (None, {'Non-teaching': {'teach_minor': 0, 'teach_major': 0}, 'Minor': {'teach_minor': 1, 'teach_major': 0},
                               'Major': {'teach_minor': 0, 'teach_major': 1}}),
    'Census region': (None, {'South': {'region_northeast': 0, 'region_midwest': 0, 'region_west': 0},
                             'Northeast': {'region_northeast': 1, 'region_midwest': 0, 'region_west': 0},
                             'Midwest': {'region_northeast': 0, 'region_midwest': 1, 'region_west': 0},
                             'West': {'region_northeast': 0, 'region_midwest': 0, 'region_west': 1}}),
    'Medicaid expansion': ('expansion', {'No': {'expansion': 0}, 'Yes': {'expansion': 1}}),
}


def design(X, interaction=True):
    Xd = X.copy()
    if interaction and 'cah' in Xd and 'expansion' in Xd:
        Xd['cah_x_expansion'] = Xd['cah'] * Xd['expansion']
    return sm.add_constant(Xd, has_constant='add')


def fit_beta(y, Xd, groups):
    mod = BetaModel(np.asarray(y, float), Xd)
    res = mod.fit(method='bfgs', maxiter=5000, disp=False, cov_type='cluster', cov_kwds={'groups': groups})
    if not res.mle_retvals.get('converged', True):
        res = mod.fit(method='newton', maxiter=200, disp=False, cov_type='cluster', cov_kwds={'groups': groups},
                      start_params=res.params)
    return res


def mean_part(res, Xd):
    k = Xd.shape[1]
    return np.asarray(res.params)[:k], np.asarray(res.cov_params())[:k, :k]


def _set_levels(Xd, settings):
    Z = Xd.copy()
    for c, v in settings.items():
        if c in Z:
            Z[c] = v
    if 'cah_x_expansion' in Z:
        Z['cah_x_expansion'] = Z['cah'] * Z['expansion']
    return Z


def standardized_means(beta, Xd, settings_by_level):
    """Adjusted mean UCR for each level (all hospitals set to that level) and its gradient w.r.t. the mean parameters."""
    est, grad = {}, {}
    for lvl, s in settings_by_level.items():
        Z = _set_levels(Xd, s).values
        mu = expit(Z @ beta)
        est[lvl] = mu.mean()
        grad[lvl] = ((mu * (1 - mu))[:, None] * Z).mean(axis=0)
    return est, grad


def contrast_rows(beta, V, Xd, name, levels):
    est, grad = standardized_means(beta, Xd, levels)
    ref = list(levels)[0]
    rows = []
    for lvl in levels:
        g_m = grad[lvl]
        se_m = np.sqrt(g_m @ V @ g_m)
        row = {'variable': name, 'level': lvl, 'adj_mean': est[lvl], 'adj_mean_se': se_m}
        if lvl != ref:
            gd = grad[lvl] - grad[ref]
            gr = grad[lvl] / est[lvl] - grad[ref] / est[ref]
            row.update({'ame': est[lvl] - est[ref], 'ame_se': np.sqrt(gd @ V @ gd),
                        'log_ratio': np.log(est[lvl] / est[ref]), 'log_ratio_se': np.sqrt(gr @ V @ gr)})
        rows.append(row)
    return rows


def interaction_rows(beta, V, Xd):
    cells = {'Non-CAH, non-expansion': {'cah': 0, 'expansion': 0}, 'Non-CAH, expansion': {'cah': 0, 'expansion': 1},
             'CAH, non-expansion': {'cah': 1, 'expansion': 0}, 'CAH, expansion': {'cah': 1, 'expansion': 1}}
    est, grad = standardized_means(beta, Xd, cells)
    rows = [{'variable': 'CAH x Medicaid expansion', 'level': k, 'adj_mean': est[k], 'adj_mean_se': np.sqrt(grad[k] @ V @ grad[k])} for k in cells]
    for e, (a, b) in {'non-expansion': ('CAH, non-expansion', 'Non-CAH, non-expansion'), 'expansion': ('CAH, expansion', 'Non-CAH, expansion')}.items():
        gd = grad[a] - grad[b]
        rows.append({'variable': 'CAH x Medicaid expansion', 'level': f'CAH difference within {e} states', 'ame': est[a] - est[b],
                     'ame_se': np.sqrt(gd @ V @ gd)})
    gdd = (grad['CAH, expansion'] - grad['Non-CAH, expansion']) - (grad['CAH, non-expansion'] - grad['Non-CAH, non-expansion'])
    dd = (est['CAH, expansion'] - est['Non-CAH, expansion']) - (est['CAH, non-expansion'] - est['Non-CAH, non-expansion'])
    rows.append({'variable': 'CAH x Medicaid expansion', 'level': 'Interaction, additive scale (difference in differences)', 'ame': dd,
                 'ame_se': np.sqrt(gdd @ V @ gdd)})
    return rows


def finalize(rows, df_resid=None):
    t = pd.DataFrame(rows)
    z = stats.norm.ppf(0.975) if df_resid is None else stats.t.ppf(0.975, df_resid)
    for est, se, lo, hi in [('adj_mean', 'adj_mean_se', 'adj_mean_lo', 'adj_mean_hi'), ('ame', 'ame_se', 'ame_lo', 'ame_hi')]:
        if est in t:
            t[lo], t[hi] = t[est] - z * t[se], t[est] + z * t[se]
    if 'log_ratio' in t:
        t['ratio'] = np.exp(t['log_ratio'])
        t['ratio_lo'] = np.exp(t['log_ratio'] - z * t['log_ratio_se'])
        t['ratio_hi'] = np.exp(t['log_ratio'] + z * t['log_ratio_se'])
    # P values: Wald test of the log ratio where a ratio is reported (so that the P value agrees with the ratio's CI), otherwise of
    # the difference in adjusted means (eg, the CAH differences and the additive interaction)
    zz = pd.Series(np.nan, index=t.index)
    if 'ame' in t:
        zz = t['ame'] / t['ame_se']
    if 'log_ratio' in t:
        zz = (t['log_ratio'] / t['log_ratio_se']).where(t['log_ratio'].notna(), zz)
    if zz.notna().any():
        t['p'] = 2 * (stats.norm.sf(zz.abs()) if df_resid is None else stats.t.sf(zz.abs(), df_resid))
    return t


def coef_table(params, cov, names, df_resid=None):
    se = np.sqrt(np.diag(cov))
    z = params / se
    crit = stats.norm.ppf(0.975) if df_resid is None else stats.t.ppf(0.975, df_resid)
    p = 2 * (stats.norm.sf(np.abs(z)) if df_resid is None else stats.t.sf(np.abs(z), df_resid))
    return pd.DataFrame({'term': names, 'coef': params, 'se': se, 'OR': np.exp(params), 'OR_lo': np.exp(params - crit * se),
                         'OR_hi': np.exp(params + crit * se), 'p': p})


def pseudo_r2(res, y, Xd):
    beta, _ = mean_part(res, Xd)
    return float(np.corrcoef(np.log(y / (1 - y)), Xd.values @ beta)[0, 1] ** 2)


def effects(beta, V, Xd, contrasts=SET_A_CONTRASTS, interaction=True):
    rows = []
    for name, (_, levels) in contrasts.items():
        # keep levels that are representable in this design (a level needing an absent indicator = 1 is not present)
        lv = {k: s for k, s in levels.items() if all(c in Xd or v == 0 for c, v in s.items())}
        if len(lv) >= 2 and any(c in Xd for s in lv.values() for c in s):
            rows += contrast_rows(beta, V, Xd, name, lv)
    if interaction and 'cah_x_expansion' in Xd:
        rows += interaction_rows(beta, V, Xd)
    return rows


# ---------------------------------------------------------------------------------------------- Rubin's rules
def rubin(estimates, variances):
    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = q.shape[0]
    qbar, ubar, b = q.mean(axis=0), u.mean(axis=0), q.var(axis=0, ddof=1)
    t = ubar + (1 + 1 / m) * b
    return qbar, t


def fit_mi(y, X_list, groups, interaction=True):
    """Fit the beta regression on each imputed data set and pool coefficients and adjusted effects with Rubin's rules."""
    coefs, covs, eff_est, eff_var, names, keys = [], [], [], [], None, None
    for X in X_list:
        Xd = design(X, interaction)
        res = fit_beta(y, Xd, groups)
        beta, V = mean_part(res, Xd)
        coefs.append(beta)
        covs.append(np.diag(V))
        names = list(Xd.columns)
        e = pd.DataFrame(effects(beta, V, Xd, interaction=interaction))
        keys = e[['variable', 'level']]
        est_cols = [c for c in ['adj_mean', 'ame', 'log_ratio'] if c in e]
        eff_est.append(e[est_cols].values.astype(float))
        eff_var.append((e[[c + '_se' for c in est_cols]].values.astype(float)) ** 2)
    qbar, tvar = rubin(coefs, covs)
    ct = coef_table(qbar, np.diag(tvar), names)
    e_q, e_t = rubin(eff_est, eff_var)
    eff = keys.copy()
    for j, c in enumerate(est_cols):
        eff[c], eff[c + '_se'] = e_q[:, j], np.sqrt(e_t[:, j])
    return ct, finalize(eff.to_dict('records'))


# ---------------------------------------------------------------------------------------------- analyses
def vif_table(X):
    Xc = sm.add_constant(X.dropna())
    return pd.DataFrame({'term': X.columns, 'VIF': [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]})


def run(d, S, zero_frame=None, S_zero=None, mi_sets=None):
    groups = pd.factorize(d['st'])[0]
    y = d['ucr'].values
    out = {}

    # M1: core set + CAH x expansion (no missing data)
    Xd = design(S['A'])
    r1 = fit_beta(y, Xd, groups)
    b1, V1 = mean_part(r1, Xd)
    m1_eff = finalize(effects(b1, V1, Xd))
    m1_coef = coef_table(np.asarray(r1.params), np.asarray(r1.cov_params()), list(Xd.columns) + ['precision'])
    save_df(m1_eff, 'tables/beta_M1_adjusted.csv')
    save_df(m1_coef, 'tables/beta_M1_coef.csv')
    save_df(vif_table(S['A']), 'tables/beta_M1_vif.csv')
    out['M1'] = {'n': int(len(y)), 'clusters': int(len(np.unique(groups))), 'pseudo_r2': pseudo_r2(r1, y, Xd),
                 'converged': bool(r1.mle_retvals.get('converged', True)), 'interaction_or': float(np.exp(b1[list(Xd.columns).index('cah_x_expansion')]))}
    log('   M1:', out['M1'])

    # M2 / M3: extended sets with multiple imputation
    for key, setname in [('M2', 'B'), ('M3', 'C')]:
        Xs = mi_sets[setname]
        ct, eff = fit_mi(y, Xs, groups)
        added = [t for t in ct.term if t not in list(Xd.columns)]
        mask = ct.term.isin(added)
        ct.loc[mask, 'q_bh'] = multipletests(ct.loc[mask, 'p'], method='fdr_bh')[1]
        save_df(ct, f'tables/beta_{key}_coef.csv')
        save_df(eff, f'tables/beta_{key}_adjusted.csv')
        out[key] = {'n_imputations': len(Xs), 'added_terms_q_lt_0.05': ct.loc[mask & (ct.q_bh < 0.05), 'term'].tolist()}
        log(f'   {key}:', out[key])

    # M4: stratified by CAH status (core set without CAH; sparse levels dropped)
    rows4 = []
    for lab, sel in [('CAH', d['cah'].values == 1), ('Non-CAH', d['cah'].values == 0)]:
        Xs = S['A'].loc[sel].drop(columns=['cah'])
        keep = [c for c in Xs if min((Xs[c] == 1).sum(), (Xs[c] == 0).sum()) >= 10]
        Xs = Xs[keep]
        Xds = design(Xs, interaction=False)
        rs = fit_beta(y[sel], Xds, pd.factorize(d.loc[sel, 'st'])[0])
        bs, Vs = mean_part(rs, Xds)
        e = finalize(effects(bs, Vs, Xds, interaction=False))
        e.insert(0, 'stratum', lab)
        rows4.append(e)
        out[f'M4_{lab}'] = {'n': int(sel.sum()), 'dropped_sparse_terms': [c for c in S['A'].columns if c not in keep and c != 'cah']}
    save_df(pd.concat(rows4), 'tables/beta_M4_stratified_adjusted.csv')

    # Sensitivity analyses
    sens = []
    def add_sens(label, eff):
        e = eff[eff['ame'].notna()].copy() if 'ame' in eff else eff.copy()
        e.insert(0, 'analysis', label)
        sens.append(e)

    add_sens('Primary model (M1)', m1_eff)
    # (a) secondary outcome measured at cost: S-10 line 30 / total operating expenses
    ok = d['ucr30_opex'].notna().values
    Xd_a = design(S['A'].loc[ok])
    ra = fit_beta(d.loc[ok, 'ucr30_opex'].values, Xd_a, pd.factorize(d.loc[ok, 'st'])[0])
    ba, Va = mean_part(ra, Xd_a)
    add_sens('Outcome at cost: line 30 / operating expenses', finalize(effects(ba, Va, Xd_a)))
    # (b) zero-inclusive frame, fractional logit (Papke-Wooldridge) with state-clustered SEs
    if zero_frame is not None:
        Xz = design(S_zero['A'])
        gz = pd.factorize(zero_frame['st'])[0]
        fz = sm.GLM(zero_frame['ucr'].clip(lower=0).values, Xz, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': gz})
        add_sens(f'Including hospitals reporting zero (fractional logit, n={len(zero_frame)})',
                 finalize(effects(np.asarray(fz.params), np.asarray(fz.cov_params()), Xz)))
        Xm = design(S['A'])
        fm = sm.GLM(y, Xm, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': groups})
        add_sens('Fractional logit, analytic sample', finalize(effects(np.asarray(fm.params), np.asarray(fm.cov_params()), Xm)))
    # (c) excluding states with expansion transitions in 2023 (SD, NC) or partial coverage (WI)
    keep_c = d['exp_transition_or_wi'].values == 0
    Xd_c = design(S['A'].loc[keep_c])
    rc = fit_beta(y[keep_c], Xd_c, pd.factorize(d.loc[keep_c, 'st'])[0])
    bc, Vc = mean_part(rc, Xd_c)
    add_sens('Excluding SD, NC and WI', finalize(effects(bc, Vc, Xd_c)))
    # (d) respondents with complete social-needs data (complete-case extended model)
    Xb = S['B']
    cc = Xb.notna().all(axis=1).values
    Xd_d = design(Xb.loc[cc])
    rd = fit_beta(y[cc], Xd_d, pd.factorize(d.loc[cc, 'st'])[0])
    bd, Vd = mean_part(rd, Xd_d)
    add_sens(f'Extended model, complete cases (n={int(cc.sum())})', finalize(effects(bd, Vd, Xd_d)))
    ct_cc = coef_table(bd, Vd, list(Xd_d.columns))
    save_df(ct_cc, 'tables/beta_M2_complete_case_coef.csv')
    save_df(pd.concat(sens), 'tables/beta_sensitivity.csv')
    save_json(out, 'results/inferential_summary.json')
    return out


def delta_adjustment(d, S, mi_list, deltas=(-0.2, -0.1, 0.1, 0.2), m=10):
    """MNAR sensitivity: shift imputed social-needs domain scores of hospitals without observed values by delta."""
    groups = pd.factorize(d['st'])[0]
    y = d['ucr'].values
    comp = [c for c in S['B'].columns if c.startswith('sdoh_') and c != 'sdoh_screen_level']
    miss = S['B'][comp].isna()
    core = list(design(S['A']).columns)
    rows = []
    for delta in deltas:
        Xs = []
        for X in mi_list[:m]:
            Z = X.copy()
            for c in comp:
                Z.loc[miss[c], c] = (Z.loc[miss[c], c] + delta).clip(0, 1)
            Xs.append(Z)
        ct, eff = fit_mi(y, Xs, groups)
        mask = ~ct.term.isin(core)                          # same false discovery rate family as model 2
        ct.loc[mask, 'q_bh'] = multipletests(ct.loc[mask, 'p'], method='fdr_bh')[1]
        for c in comp + ['cah', 'expansion', 'own_forprofit', 'own_government']:
            r = ct[ct.term == c].iloc[0]
            rows.append({'delta': delta, 'term': c, 'coef': r['coef'], 'se': r['se'], 'p': r['p'], 'q_bh': r.get('q_bh', np.nan)})
    t = pd.DataFrame(rows)
    save_df(t, 'tables/beta_M2_delta_adjustment.csv')
    return t
