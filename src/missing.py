"""Step 3. Missing data: per-variable missingness by type, the missingness mechanism (Little's test and non-response
models), and multiple imputation by chained equations for the inferential models."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.imputation.mice import MICEData

from src import parameters as config
from src import features as F
from src.io_utils import log, save_df

warnings.filterwarnings('ignore', category=RuntimeWarning)


def missingness_table(d):
    """One row per candidate predictor with the number missing by type and the handling rule (Multimedia Appendix)."""
    items = F.sdoh_items(config.VARLIST_CSV)
    resp = d['respondent'].values == 1
    block_missing = d[items].isna().all(axis=1).values
    no_screen = (d[F.SCREEN_GATE] == 'No').values
    skip_items = set([c for c in items if c.endswith(' screening')] + ['Does hospital or system record social needs screening results in EHR?'])
    rows = []

    def add(var, block, miss, not_applicable, rule):
        miss = np.asarray(miss, bool)
        na = np.asarray(not_applicable, bool) & miss
        unit = miss & ~resp & ~na
        section = miss & resp & block_missing & ~na if block == 'Social needs and health equity' else np.zeros_like(miss)
        item = miss & ~na & ~unit & ~section
        n = len(miss)
        rows.append({'variable': var, 'block': block, 'missing_n': int(miss.sum()), 'missing_pct': round(100 * miss.mean(), 1),
                     'not_applicable_n': int(na.sum()), 'unit_nonresponse_n': int(unit.sum()), 'section_nonresponse_n': int(section.sum()),
                     'item_nonresponse_n': int(item.sum()), 'n': n, 'handling': rule})

    for c in ['Critical Access Hospital', 'Control / Ownership Type', 'Hospital size', 'Hospital Location', 'Teaching Status', 'State']:
        add(c, 'Core', d[c].isna(), np.zeros(len(d)), 'Complete')
    add('System member', 'Core', np.zeros(len(d), bool), np.zeros(len(d)),
        f'Complete: the AHA field is blank for the {int(d["System ID"].isna().sum()):,} hospitals without a system identifier, which are not system members (coded 0)')
    add('Degree of centralization of health system', 'Extended', d['Degree of centralization of health system'].isna(), d['System ID'].isna(),
        'Not applicable to non-system hospitals (reference category)')
    for c, _ in F.STRUCT_YN:
        m = d[c].isna()
        add(c, 'Extended', m, np.zeros(len(d)), 'Complete' if m.sum() == 0 else 'Multiple imputation (inferential models); learned missing-value handling (machine learning)')
    for c in ['Emergency room visits', 'Total outpatient visits']:
        add(c, 'Extended', d[c].isna(), np.zeros(len(d)), f'Complete; estimated by AHA for {int(d.utilization_estimated.sum())} hospitals (flag included)')
    for c in items:
        rule = ('Not applicable when the hospital does not screen for social needs (coded 0); otherwise as below' if c in skip_items else
                'Summarized in domain scores; multiple imputation (inferential models); learned missing-value handling (machine learning)')
        add(c, 'Social needs and health equity', d[c].isna(), no_screen if c in skip_items else np.zeros(len(d)), rule)
    for c in ['sahie2023_pctui_u65', 'saipe2023_pov_pct_all', 'saipe2023_mhi', 'laus2023_unemp_rate']:
        add(c, 'County', d[c].isna(), np.zeros(len(d)), 'Linked by county FIPS')
    for c in ['medicaid_day_share', 'selfpay_discharge_share']:
        add(c, 'Payer mix', d[c].isna(), np.zeros(len(d)), f'Estimated by AHA for {int(d.payer_mix_estimated.sum())} hospitals (flag included); multiple imputation if missing')
    # derived model variables
    X = F.build_sets(d, config.VARLIST_CSV)['C']
    add('Micropolitan area (from CBSA type)', 'Extended', X['cbsa_micro'].isna(), np.zeros(len(d)), 'Complete')
    add('Medicaid expansion state (KFF)', 'Core', X['expansion'].isna(), np.zeros(len(d)), 'Complete (linked by state)')
    for c in ['aha_estimated_utilization', 'pm_aha_estimated']:
        add(F.LABELS[c], 'Data provenance flags', X[c].isna(), np.zeros(len(d)), 'Complete')
    for c in [c for c in X.columns if c.startswith('sdoh_')]:
        add(F.LABELS[c], 'Social needs domain scores (derived)', X[c].isna(), np.zeros(len(d)),
            'Missing if <80% of the domain items were answered; multiple imputation (inferential models); learned missing-value handling (machine learning)')
    t = pd.DataFrame(rows)
    save_df(t, 'tables/missingness_by_variable.csv')
    sd = t[t.block == 'Social needs and health equity']
    summary = {
        'n': len(d), 'unit_nonresponse_n': int((~resp).sum()), 'unit_nonresponse_pct': round(100 * (~resp).mean(), 1),
        'section_nonresponse_n': int((resp & block_missing).sum()), 'section_nonresponse_pct': round(100 * (resp & block_missing).mean(), 1),
        'sdoh_block_unavailable_n': int(block_missing.sum()), 'sdoh_block_unavailable_pct': round(100 * block_missing.mean(), 1),
        'non_screeners_n': int(no_screen.sum()), 'sdoh_item_missing_pct_range': [float(sd.missing_pct.min()), float(sd.missing_pct.max())],
        'item_nonresponse_pct_of_section_respondents_range': [
            round(100 * sd.item_nonresponse_n.min() / max(1, (resp & ~block_missing).sum()), 1),
            round(100 * sd.item_nonresponse_n.max() / max(1, (resp & ~block_missing).sum()), 1)],
    }
    log('   missingness summary:', summary)
    return t, summary, block_missing


def nonresponse_models(d, block_missing):
    """Logistic models of (1) AHA survey non-response and (2) unavailable social-needs section on fully observed
    hospital characteristics, with and without the outcome (logit UCR); standard errors clustered by state."""
    x = d.copy()
    x['logit_ucr'] = np.log(x['ucr'] / (1 - x['ucr']))
    x['unit_nr'] = (x['respondent'] == 0).astype(int)
    x['block_nr'] = block_missing.astype(int)
    x['own_c'] = pd.Categorical(x['own'], ['Not-for-profit', 'For-profit', 'Government'])
    x['size_c'] = pd.Categorical(x['size'], ['Small', 'Medium', 'Large'])
    x['teach_c'] = pd.Categorical(x['teach'], ['Non Teaching', 'Minor', 'Major'])
    x['region_c'] = pd.Categorical(x['region'], ['South', 'Northeast', 'Midwest', 'West'])
    g = pd.factorize(x['st'])[0]
    rhs = 'cah + C(own_c) + C(size_c) + rural + system + C(teach_c) + C(region_c) + expansion'
    rows, aucs = [], {}
    for dep, lab in [('unit_nr', 'AHA survey non-response'), ('block_nr', 'Social needs section unavailable')]:
        m0 = smf.logit(f'{dep} ~ {rhs}', data=x).fit(disp=False, cov_type='cluster', cov_kwds={'groups': g})
        m1 = smf.logit(f'{dep} ~ {rhs} + logit_ucr', data=x).fit(disp=False, cov_type='cluster', cov_kwds={'groups': g})
        aucs[lab] = {'auc_structural': roc_auc_score(x[dep], m0.predict(x)), 'auc_with_outcome': roc_auc_score(x[dep], m1.predict(x)),
                     'prevalence': float(x[dep].mean()), 'lr_wald_p_structural': float(m0.llr_pvalue)}
        ci = m1.conf_int()
        for term in m1.params.index.drop('Intercept'):
            rows.append({'model': lab, 'term': term, 'OR': np.exp(m1.params[term]), 'lo': np.exp(ci.loc[term, 0]),
                         'hi': np.exp(ci.loc[term, 1]), 'p': m1.pvalues[term]})
    t = pd.DataFrame(rows)
    save_df(t, 'tables/nonresponse_models.csv')
    log('   non-response models:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in aucs.items()})
    log('   OR per logit(UCR):', t[t.term == 'logit_ucr'][['model', 'OR', 'lo', 'hi', 'p']].round(3).to_dict('records'))
    return t, aucs


def _em_mvn(X, max_iter=500, tol=1e-8):
    n, p = X.shape
    M = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    Sig = np.cov(np.where(M, mu, X), rowvar=False, bias=True) + 1e-6 * np.eye(p)
    pats, inv = np.unique(M, axis=0, return_inverse=True)
    inv = inv.ravel()
    for it in range(max_iter):
        T1, T2 = np.zeros(p), np.zeros((p, p))
        for k, pat in enumerate(pats):
            rows = np.where(inv == k)[0]
            o, mm = ~pat, pat
            Xk = X[rows].copy()
            Ck = np.zeros((p, p))
            if mm.any() and o.any():
                B = np.linalg.solve(Sig[np.ix_(o, o)], Sig[np.ix_(mm, o)].T).T
                Xk[:, mm] = mu[mm] + (Xk[:, o] - mu[o]) @ B.T
                Ck[np.ix_(mm, mm)] = Sig[np.ix_(mm, mm)] - B @ Sig[np.ix_(mm, o)].T
            elif mm.any():
                Xk[:, mm] = mu[mm]
                Ck[np.ix_(mm, mm)] = Sig[np.ix_(mm, mm)]
            T1 += Xk.sum(axis=0)
            T2 += Xk.T @ Xk + len(rows) * Ck
        mu_new = T1 / n
        Sig_new = T2 / n - np.outer(mu_new, mu_new)
        done = np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(Sig_new - Sig)) < tol
        mu, Sig = mu_new, Sig_new
        if done:
            break
    return mu, Sig, pats, inv


def little_mcar(X):
    """Little (1988) chi-square test of missing completely at random, using EM maximum-likelihood estimates."""
    X = np.asarray(X, float)
    mu, Sig, pats, inv = _em_mvn(X)
    d2, df = 0.0, 0
    for k, pat in enumerate(pats):
        o = ~pat
        if not o.any():
            continue
        rows = np.where(inv == k)[0]
        diff = X[rows][:, o].mean(axis=0) - mu[o]
        d2 += len(rows) * diff @ np.linalg.solve(Sig[np.ix_(o, o)], diff)
        df += int(o.sum())
    df -= X.shape[1]
    return {'chi2': float(d2), 'df': int(df), 'p': float(stats.chi2.sf(d2, df)), 'patterns': int(len(pats))}


def run_mechanism(d, S):
    t, summary, block_missing = missingness_table(d)
    nr, aucs = nonresponse_models(d, block_missing)
    A, B = S['A'], S['B']
    comp = [c for c in B.columns if c.startswith('sdoh_')]
    mc = pd.concat([A[['cah', 'own_forprofit', 'own_government', 'rural', 'system', 'expansion']],
                    np.log(d['ucr'] / (1 - d['ucr'])).rename('logit_ucr'), B[comp]], axis=1).astype(float)
    lt = little_mcar(mc.values)
    lt_resp = little_mcar(mc[d['respondent'].values == 1].values)
    log('   Little MCAR test:', lt, '| respondents only:', lt_resp)
    # How the original "missing = No" coding changes associations between activity scores and the outcome
    Xno = F.item_features(d, F.sdoh_items(config.VARLIST_CSV), missing_rule='no')
    Cno, Cobs = F.composites(Xno), S['B'][comp]
    cmp_rows = []
    for c in comp:
        r0, p0 = stats.spearmanr(Cno[c], d['ucr'], nan_policy='omit')
        ok = Cobs[c].notna()
        r1, p1 = stats.spearmanr(Cobs.loc[ok, c], d.loc[ok, 'ucr'])
        cmp_rows.append({'domain': c, 'rho_missing_coded_no': r0, 'p_missing_coded_no': p0, 'rho_observed_only': r1, 'p_observed_only': p1,
                         'n_observed': int(ok.sum())})
    save_df(pd.DataFrame(cmp_rows), 'tables/missing_coded_no_vs_observed.csv')
    return {'summary': summary, 'little': lt, 'little_respondents': lt_resp, 'nonresponse_auc': aucs}


# ---------------------------------------------------------------------------------------------- multiple imputation
def impute(X, outcome, m=None, cycles=None, seed=config.SEED, by=None):
    """MICE with predictive mean matching (statsmodels). X: predictors with missing values; outcome (logit UCR) is used as an
    auxiliary variable. Imputation is run separately within the strata given by `by` (CAH vs non-CAH) to preserve
    interactions with CAH status. Returns a list of m completed copies of X."""
    m = m or config.M_IMPUTATIONS
    cycles = cycles or config.MICE_CYCLES
    groups = [np.arange(len(X))] if by is None else [np.where(by == v)[0] for v in np.unique(by)]
    out = [X.copy() for _ in range(m)]
    for gi, idx in enumerate(groups):
        Z = X.iloc[idx].copy()
        Z['aux_outcome'] = np.asarray(outcome)[idx]
        const = [c for c in Z.columns if Z[c].nunique(dropna=True) <= 1]
        Zi = Z.drop(columns=const).reset_index(drop=True)
        miss_cols = [c for c in Zi.columns if Zi[c].isna().any()]
        if not miss_cols:
            continue
        rng_seed = seed + 1000 * gi
        np.random.seed(rng_seed)
        imp = MICEData(Zi, k_pmm=5)
        imp.update_all(cycles)               # burn-in
        for j in range(m):
            imp.update_all(5)                # thin between retained data sets
            done = imp.data
            for c in miss_cols:
                if c in X.columns:
                    out[j].iloc[idx, out[j].columns.get_loc(c)] = done[c].values
    return out
