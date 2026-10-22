"""Step 5. Machine learning: repeated nested cross-validation for the continuous ratio (regression) and the high-burden
indicator (classification), with prespecified benchmarks.

Outer loop: repeated stratified 5-fold CV (strata = UCR quartile x CAH status); the same folds are used for every model, so
comparisons are paired. Inner loop: 5-fold randomized hyperparameter search on the outer-training data only. For the
classifiers, the decision threshold is the one maximizing the F1 score of the high-burden class in inner-CV predictions on
the outer-training data. All reported performance comes from the outer test folds.
"""
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.othermod.betareg import BetaModel

from src import parameters as config
from src import metrics as M
from src.io_utils import log, out_path, save_df

warnings.filterwarnings('ignore')
expit = lambda z: 1 / (1 + np.exp(-z))

LR5 = ['cah', 'own_forprofit', 'own_government', 'size_medium', 'size_large', 'rural', 'region_northeast', 'region_midwest', 'region_west']
SET_A = ['cah', 'own_forprofit', 'own_government', 'size_medium', 'size_large', 'rural', 'system', 'teach_minor', 'teach_major',
         'region_northeast', 'region_midwest', 'region_west', 'expansion']
BENCH_COLS = {'lr_cah': ['cah'], 'lr5': LR5, 'lr5_exp': LR5 + ['expansion'], 'lr_a': SET_A, 'beta_a': SET_A, 'null': SET_A[:1]}


# ---------------------------------------------------------------------------------------------- estimators
class XGBES(BaseEstimator):
    """XGBoost with early stopping on a validation split carved from the training data of the current fit."""
    _task = 'clf'

    def __init__(self, learning_rate=0.05, max_depth=4, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, n_estimators=5000, early_stopping_rounds=100,
                 validation_fraction=0.15, random_state=0, n_jobs=1):
        self.learning_rate = learning_rate; self.max_depth = max_depth; self.min_child_weight = min_child_weight
        self.subsample = subsample; self.colsample_bytree = colsample_bytree; self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha; self.gamma = gamma; self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds; self.validation_fraction = validation_fraction
        self.random_state = random_state; self.n_jobs = n_jobs

    def _model(self):
        p = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                 min_child_weight=self.min_child_weight, subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                 reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, gamma=self.gamma, tree_method='hist',
                 early_stopping_rounds=self.early_stopping_rounds, random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0)
        return (xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **p) if self._task == 'clf'
                else xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **p))

    def fit(self, X, y):
        strat = y if self._task == 'clf' else None
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.validation_fraction, stratify=strat, random_state=self.random_state)
        self.model_ = self._model()
        self.model_.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        if self._task == 'clf':
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def shap_values(self, X):
        """Exact path-dependent TreeSHAP values computed by XGBoost (log-odds scale for classification)."""
        booster = self.model_.get_booster()
        dm = xgb.DMatrix(pd.DataFrame(X).astype(float), missing=np.nan)
        it = (0, self.model_.best_iteration + 1)
        contrib = booster.predict(dm, pred_contribs=True, iteration_range=it)
        return contrib[:, :-1], contrib[:, -1]


class XGBClf(ClassifierMixin, XGBES):
    _task = 'clf'


class XGBReg(RegressorMixin, XGBES):
    _task = 'reg'


class BetaRegressor(RegressorMixin, BaseEstimator):
    """Beta regression (logit link) used as a benchmark predictor: prediction = fitted mean."""

    def __init__(self, interaction=True):
        self.interaction = interaction

    def _design(self, X):
        X = pd.DataFrame(X).copy()
        if self.interaction and 'cah' in X and 'expansion' in X:
            X['cah_x_expansion'] = X['cah'] * X['expansion']
        return np.column_stack([np.ones(len(X)), X.values.astype(float)])

    def fit(self, X, y):
        Xd = self._design(X)
        keep = Xd.std(axis=0) > 0
        keep[0] = True
        self.keep_ = keep
        res = BetaModel(np.asarray(y, float), Xd[:, keep]).fit(method='bfgs', maxiter=5000, disp=False)
        self.beta_ = np.asarray(res.params)[:keep.sum()]
        return self

    def predict(self, X):
        return expit(self._design(X)[:, self.keep_] @ self.beta_)


class MeanRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _prep():
    return [('imp', SimpleImputer(strategy='median', add_indicator=True)), ('sc', StandardScaler())]


def learner(name, task, seed):
    """Estimator, search space and number of random draws for each learner."""
    n_iter = config.N_ITER.get(name, 0)
    if name == 'xgb':
        est = XGBClf(random_state=seed) if task == 'clf' else XGBReg(random_state=seed)
        space = {'learning_rate': loguniform(0.005, 0.3), 'max_depth': randint(2, 9), 'min_child_weight': loguniform(1, 100),
                 'subsample': uniform(0.5, 0.5), 'colsample_bytree': uniform(0.3, 0.7), 'reg_lambda': loguniform(1e-3, 100),
                 'reg_alpha': loguniform(1e-4, 10), 'gamma': loguniform(1e-4, 10) if task == 'clf' else loguniform(1e-8, 1e-3)}
        if task == 'reg':
            space['reg_alpha'] = loguniform(1e-8, 1e-2)
        return est, space, n_iter
    if name == 'rf':
        est = (RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed) if task == 'clf'
               else RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed))
        space = {'max_features': uniform(0.05, 0.6), 'min_samples_leaf': randint(1, 61), 'max_depth': [None, 4, 6, 8, 12, 16],
                 'max_samples': uniform(0.5, 0.5)}
        return est, space, n_iter
    if name == 'mlp':
        mlp_space = {'hidden_layer_sizes': [(16,), (32,), (64,), (128,), (32, 16), (64, 32), (128, 64)], 'activation': ['relu', 'tanh'],
                     'alpha': loguniform(1e-6, 1e-1), 'learning_rate_init': loguniform(1e-4, 1e-2), 'batch_size': [32, 64, 128, 256]}
        common = dict(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, max_iter=2000, random_state=seed)
        if task == 'clf':
            est = Pipeline(_prep() + [('m', MLPClassifier(**common))])
            return est, {f'm__{k}': v for k, v in mlp_space.items()}, n_iter
        est = TransformedTargetRegressor(regressor=Pipeline(_prep() + [('m', MLPRegressor(**common))]), transformer=StandardScaler())
        return est, {f'regressor__m__{k}': v for k, v in mlp_space.items()}, n_iter
    if name == 'en':
        if task == 'clf':
            est = Pipeline(_prep() + [('m', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, random_state=seed))])
            return est, {'m__C': loguniform(1e-4, 1e2), 'm__l1_ratio': uniform(0, 1)}, n_iter
        est = Pipeline(_prep() + [('m', ElasticNet(max_iter=20000, random_state=seed))])
        return est, {'m__alpha': loguniform(1e-7, 1e-1), 'm__l1_ratio': uniform(0, 1)}, n_iter
    # benchmarks (no tuning)
    if name in ('lr_cah', 'lr5', 'lr5_exp', 'lr_a'):
        return LogisticRegression(penalty=None, max_iter=5000), None, 0
    if name == 'beta_a':
        return BetaRegressor(), None, 0
    if name == 'null':
        return MeanRegressor(), None, 0
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------- folds
def outer_folds(ucr, strat_extra=None, repeats=None, seed=config.SEED):
    repeats = repeats or config.OUTER_REPEATS
    q = pd.qcut(pd.Series(ucr).rank(method='first'), 4, labels=False).values
    strat = q * 2 + (np.zeros(len(q), int) if strat_extra is None else np.asarray(strat_extra, int))
    rskf = RepeatedStratifiedKFold(n_splits=config.OUTER_FOLDS, n_repeats=repeats, random_state=seed)
    return [(i // config.OUTER_FOLDS, i % config.OUTER_FOLDS, tr, te) for i, (tr, te) in enumerate(rskf.split(np.zeros(len(q)), strat))]


# ---------------------------------------------------------------------------------------------- runner
def run_spec(spec, X, ucr, high, folds, learners, tasks=('reg', 'clf'), shap_learner='xgb', save_models=False):
    """Fit every learner on every outer fold; save fold metrics, out-of-fold predictions, selected hyperparameters and,
    for `shap_learner`, out-of-fold SHAP values."""
    X = X.reset_index(drop=True)
    ucr = np.asarray(ucr, float)
    high = np.asarray(high, int)
    rows, oof, shap_rows, params_rows = [], [], [], []
    for task in tasks:
        y = high if task == 'clf' else ucr
        inner = (StratifiedKFold(config.INNER_FOLDS, shuffle=True, random_state=config.SEED) if task == 'clf'
                 else KFold(config.INNER_FOLDS, shuffle=True, random_state=config.SEED))
        scoring = 'neg_log_loss' if task == 'clf' else 'neg_mean_squared_error'
        for lname in learners:
            if task == 'clf' and lname in ('beta_a', 'null'):
                continue
            if task == 'reg' and lname in ('lr_cah', 'lr5', 'lr5_exp', 'lr_a'):
                continue
            cols = [c for c in BENCH_COLS[lname] if c in X.columns] if lname in BENCH_COLS else list(X.columns)
            cols = [c for c in cols if X[c].nunique(dropna=True) > 1] or cols
            Xl = X[cols]
            t0 = time.time()
            for rep, k, tr, te in folds:
                seed = config.SEED + 100 * rep + k
                est, space, n_iter = learner(lname, task, seed)
                Xtr, Xte, ytr, yte = Xl.iloc[tr], Xl.iloc[te], y[tr], y[te]
                if space:
                    search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring=scoring, cv=inner, n_jobs=config.N_JOBS,
                                                random_state=seed, refit=True, error_score='raise')
                    search.fit(Xtr, ytr)
                    best, bp = search.best_estimator_, search.best_params_
                else:
                    best, bp = clone(est).fit(Xtr, ytr), {}
                params_rows.append({'spec': spec, 'task': task, 'learner': lname, 'repeat': rep, 'fold': k,
                                    **{p: (str(v) if isinstance(v, tuple) else v) for p, v in bp.items()}})
                if task == 'clf':
                    p_in = cross_val_predict(clone(best), Xtr, ytr, cv=inner, method='predict_proba', n_jobs=config.N_JOBS)[:, 1]
                    tau = M.best_f1_threshold(ytr, p_in)
                    pred = best.predict_proba(Xte)[:, 1]
                    met = M.clf_metrics(yte, pred, tau, ytr.mean())
                else:
                    pred = best.predict(Xte)
                    met = M.reg_metrics(yte, pred)
                rows.append({'spec': spec, 'task': task, 'learner': lname, 'repeat': rep, 'fold': k, 'n_train': len(tr), 'n_test': len(te), **met})
                oof.append(pd.DataFrame({'spec': spec, 'task': task, 'learner': lname, 'repeat': rep, 'fold': k, 'row': te,
                                         'y': yte, 'pred': pred}))
                if lname == shap_learner and hasattr(best, 'shap_values'):
                    sv, base = best.shap_values(Xte)
                    s = pd.DataFrame(sv, columns=cols)
                    s.insert(0, 'row', te); s.insert(0, 'fold', k); s.insert(0, 'repeat', rep); s.insert(0, 'task', task)
                    s['base_value'] = base
                    shap_rows.append(s)
            log(f'   {spec:<18s} {task} {lname:<8s} done in {time.time() - t0:6.0f}s')
    folds_df = pd.DataFrame(rows)
    save_df(folds_df, f'tables/ml_folds_{spec}.csv')
    pd.concat(oof).to_parquet(out_path(f'data/oof_{spec}.parquet'))
    if params_rows:
        save_df(pd.DataFrame(params_rows), f'tables/ml_selected_hyperparameters_{spec}.csv')
    if shap_rows:
        pd.concat(shap_rows).to_parquet(out_path(f'data/shap_{spec}.parquet'))
    return folds_df


def summarize(folds_df):
    """Mean and Nadeau-Bengio corrected 95% CI over outer folds."""
    out = []
    metrics = [c for c in folds_df.columns if c not in ('spec', 'task', 'learner', 'repeat', 'fold', 'n_train', 'n_test')]
    for (spec, task, lname), g in folds_df.groupby(['spec', 'task', 'learner'], sort=False):
        row = {'spec': spec, 'task': task, 'learner': lname, 'n_fits': len(g)}
        for mtr in metrics:
            if g[mtr].notna().any():
                mean, lo, hi = M.corrected_ci(g[mtr], g['n_train'].mean(), g['n_test'].mean())
                row[mtr], row[mtr + '_lo'], row[mtr + '_hi'] = mean, lo, hi
        for c in ['tp', 'fp', 'fn', 'tn']:
            if c in g and g[c].notna().any():
                row[c + '_sum_repeat0'] = int(g.loc[g.repeat == 0, c].sum())
        out.append(row)
    return pd.DataFrame(out)


def paired_comparison(folds_a, folds_b, metric):
    a = folds_a.set_index(['repeat', 'fold'])[metric]
    b = folds_b.set_index(['repeat', 'fold'])[metric]
    j = a.index.intersection(b.index)
    d = (a.loc[j] - b.loc[j]).values
    n_tr, n_te = folds_a['n_train'].mean(), folds_a['n_test'].mean()
    return M.corrected_ttest(d, n_tr, n_te)
