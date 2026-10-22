"""Step 8. Diagnostic for the negative XGBoost R-squared of the original submission (reviewer comment AO7).

(1) On the corrected sample (set C, first-repeat outer folds), compare an untuned default configuration, the configuration
    found in the original notebook (capped logit target, 3,000 trees, no early stopping), the tuned primary specification,
    and a tuned model on the logit scale with naive and smearing back-transformation.
(2) On the original submission's sample, show how much test-set R-squared varies across random 80/20 splits for the
    configuration reported in the original Methods.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, train_test_split
from scipy.stats import loguniform, randint, uniform

from src import parameters as config
from src import features as F
from src import metrics as M
from src import ml, sample
from src.io_utils import log, out_path, save_df

expit = lambda z: 1 / (1 + np.exp(-z))
CAP, EPS = 0.2, 1e-6


def _logit_cap(y):
    z = np.clip(y / CAP, EPS, 1 - EPS)
    return np.log(z / (1 - z))


def run(d, S):
    X = S['C'].reset_index(drop=True)
    y = d['ucr'].values
    folds_raw = json.loads(out_path('data/folds_pooled.json').read_text())
    test_sets = [np.array(te) for r, k, te in folds_raw if r == 0]
    rows = []
    for k, te in enumerate(test_sets):
        tr = np.setdiff1d(np.arange(len(y)), te)
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        # (a) default configuration, raw target
        m = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=config.SEED, n_jobs=config.N_JOBS)
        m.fit(Xtr, ytr)
        rows.append({'condition': 'a. Default XGBoost (100 trees, learning rate 0.1), raw target', 'fold': k, **M.reg_metrics(yte, m.predict(Xte))})
        # (b) configuration in the original notebook: capped logit target, 3,000 trees, no early stopping
        m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=6, min_child_weight=4, subsample=0.8, colsample_bytree=0.7,
                             reg_alpha=0.5, reg_lambda=1.0, random_state=config.SEED, n_jobs=config.N_JOBS)
        m.fit(Xtr, _logit_cap(ytr))
        rows.append({'condition': 'b. Original notebook configuration (logit of UCR/0.2, 3,000 trees, no early stopping)', 'fold': k,
                     **M.reg_metrics(yte, expit(m.predict(Xte)) * CAP)})
        # (d, e) tuned on the logit scale; naive vs smearing back-transformation
        est, _, n_iter = ml.learner('xgb', 'reg', config.SEED + k)
        space = {'learning_rate': loguniform(0.005, 0.3), 'max_depth': randint(2, 9), 'min_child_weight': loguniform(1, 100),
                 'subsample': uniform(0.5, 0.5), 'colsample_bytree': uniform(0.3, 0.7), 'reg_lambda': loguniform(1e-3, 100),
                 'reg_alpha': loguniform(1e-4, 10), 'gamma': loguniform(1e-4, 10)}
        zt = np.log(ytr / (1 - ytr))
        inner = KFold(config.INNER_FOLDS, shuffle=True, random_state=config.SEED)
        s = RandomizedSearchCV(est, space, n_iter=n_iter, scoring='neg_mean_squared_error', cv=inner, n_jobs=config.N_JOBS,
                               random_state=config.SEED + k).fit(Xtr, zt)
        eta = s.best_estimator_.predict(Xte)
        rows.append({'condition': 'd. Tuned, logit target, naive back-transformation', 'fold': k, **M.reg_metrics(yte, expit(eta))})
        resid = zt - cross_val_predict(s.best_estimator_, Xtr, zt, cv=inner, n_jobs=config.N_JOBS)
        smear = expit(eta[:, None] + resid[None, :]).mean(axis=1)
        rows.append({'condition': 'e. Tuned, logit target, smearing back-transformation', 'fold': k, **M.reg_metrics(yte, smear)})
    # (c) the primary specification (tuned on the UCR scale), same folds
    f = pd.read_csv(out_path('tables/ml_folds_pooled_C.csv'))
    f = f[(f.task == 'reg') & (f.learner == 'xgb') & (f.repeat == 0)]
    for _, r in f.iterrows():
        rows.append({'condition': 'c. Tuned (primary specification), UCR scale', 'fold': int(r['fold']),
                     **{m_: r[m_] for m_ in ['mae', 'rmse', 'r2', 'spearman', 'cal_slope']}})
    t = pd.DataFrame(rows)
    save_df(t, 'tables/ao7_diagnostic_folds.csv')
    summ = t.groupby('condition')[['r2', 'rmse', 'mae', 'spearman']].agg(['mean', 'min', 'max']).round(4)
    summ.columns = ['_'.join(c) for c in summ.columns]
    save_df(summ.reset_index(), 'tables/ao7_diagnostic.csv')
    log('   AO7 corrected sample:', summ['r2_mean'].to_dict())

    # (2) original submission's sample: variability of a single 80/20 split
    try:
        ids = pd.read_csv(config.PUBLISHED_IDS_CSV, usecols=['AHA ID'], dtype={'AHA ID': str})['AHA ID']
    except FileNotFoundError:
        return t
    o = sample.frame_for_ids(ids)
    o = o[(o['ucr'] > 0) & (o['ucr'] < config.UCR_MAX)].reset_index(drop=True)
    Xo = F.build_sets(o, config.VARLIST_CSV)['C']
    yo = o['ucr'].values
    split_rows = []
    for seed in range(100):
        tr, te = train_test_split(np.arange(len(yo)), test_size=0.2, random_state=seed)
        mx = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, reg_lambda=1, reg_alpha=0, random_state=42, n_jobs=config.N_JOBS)
        mx.fit(Xo.iloc[tr], yo[tr])
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=config.N_JOBS).fit(Xo.iloc[tr].fillna(-1), yo[tr])
        split_rows.append({'split_seed': seed, 'r2_xgb_reported_config': M.reg_metrics(yo[te], mx.predict(Xo.iloc[te]))['r2'],
                           'r2_rf_reported_config': M.reg_metrics(yo[te], rf.predict(Xo.iloc[te].fillna(-1)))['r2']})
    sp = pd.DataFrame(split_rows)
    save_df(sp, 'tables/ao7_original_sample_splits.csv')
    log(f'   AO7 original sample (n={len(yo)}): XGB R2 over 100 splits min {sp.r2_xgb_reported_config.min():.3f}, median '
        f'{sp.r2_xgb_reported_config.median():.3f}, max {sp.r2_xgb_reported_config.max():.3f}; negative in '
        f'{int((sp.r2_xgb_reported_config < 0).sum())} splits; seed 42: {sp.loc[sp.split_seed == 42, "r2_xgb_reported_config"].iloc[0]:.3f}')
    return t
