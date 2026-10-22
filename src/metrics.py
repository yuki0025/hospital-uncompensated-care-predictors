"""Performance metrics, threshold selection, corrected resampled t statistics, and DeLong's test."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)

EPS = 1e-6


def best_f1_threshold(y, p):
    """Threshold maximizing the F1 score of the positive (high-burden) class; chosen on training-fold predictions only."""
    grid = np.unique(np.round(p, 4))
    if len(grid) > 400:
        grid = np.quantile(p, np.linspace(0.01, 0.99, 400))
    f1 = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


def calibration(y, p):
    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y)
    # calibration-in-the-large: intercept with the linear predictor as an offset
    import statsmodels.api as sm
    cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return float(cil.params[0]), float(slope.coef_[0][0])


def clf_metrics(y, p, tau, prevalence_train):
    y = np.asarray(y).astype(int)
    yhat = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    brier = brier_score_loss(y, p)
    brier_null = brier_score_loss(y, np.full_like(p, prevalence_train, dtype=float))
    ci, cs = calibration(y, p)
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    rec = recall_score(y, yhat, zero_division=0)
    return {
        'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'prevalence': float(y.mean()),
        'brier': brier, 'scaled_brier': 1 - brier / brier_null, 'log_loss': log_loss(y, np.clip(p, EPS, 1 - EPS)),
        'cal_intercept': ci, 'cal_slope': cs, 'threshold': tau,
        'precision': precision_score(y, yhat, zero_division=0), 'recall': rec, 'f1': f1_score(y, yhat, zero_division=0),
        'specificity': spec, 'balanced_accuracy': (rec + spec) / 2, 'accuracy': (tp + tn) / len(y),
        'macro_precision': precision_score(y, yhat, average='macro', zero_division=0),
        'macro_recall': recall_score(y, yhat, average='macro', zero_division=0),
        'macro_f1': f1_score(y, yhat, average='macro', zero_division=0),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def reg_metrics(y, yhat):
    slope = np.polyfit(yhat, y, 1)[0] if np.std(yhat) > 0 else np.nan
    return {'mae': mean_absolute_error(y, yhat), 'rmse': float(np.sqrt(mean_squared_error(y, yhat))), 'r2': r2_score(y, yhat),
            'spearman': stats.spearmanr(y, yhat).correlation, 'cal_slope': slope}


def corrected_ci(values, n_train, n_test, alpha=0.05):
    """Mean over outer folds with the Nadeau-Bengio corrected variance."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan
    var = (1 / k + n_test / n_train) * v.var(ddof=1)
    h = stats.t.ppf(1 - alpha / 2, k - 1) * np.sqrt(var)
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def corrected_ttest(diffs, n_train, n_test):
    d = np.asarray(diffs, float)
    k = len(d)
    var = (1 / k + n_test / n_train) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    return float(d.mean()), float(2 * stats.t.sf(abs(t), k - 1))


# ---------------------------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, p1, p2):
    """DeLong test for two correlated AUROCs (Sun & Xu fast algorithm). Returns auc1, auc2, p value."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort')
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    diff_var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(diff_var) if diff_var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))
