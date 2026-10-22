"""Figure helpers (static figures for print; 300 dpi PNG). Colors: validated categorical slots (blue, orange, aqua) and a
blue-gray-red diverging scale for SHAP feature values; text in neutral inks."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e7e6e3', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
DIVERGING = LinearSegmentedColormap.from_list('lowhigh', [BLUE, '#b9b8b4', RED])
DPI = 300

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'axes.grid': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False,
})


def _grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- flow diagram
def flow_diagram(flow, final_split, path):
    """Vertical flow with one side box per exclusion step, ending in a split by CAH status."""
    import textwrap
    steps = flow.to_dict('records')
    n = len(steps)
    gap = 1.55
    fig, ax = plt.subplots(figsize=(7.6, gap * 0.78 * n + 2.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-(gap * (n - 1) + 2.6), 0.75)
    ax.axis('off')
    xm, wm = 3.1, 5.2           # main column
    xs, ws = 9.35, 5.7          # side column

    def box(x, y, w, h, text, bold=False, fc='#f5f5f3'):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    linewidth=0.9, edgecolor=INK2, facecolor=fc))
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold' if bold else 'normal', linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, shrinkA=0, shrinkB=0))

    wrap = lambda t, w: '\n'.join(textwrap.wrap(t, w))
    ys = [-gap * i for i in range(n)]
    hm = 0.68
    mtxts = [wrap(st['main_label'], 38) + f"\n(n = {st['n_remaining']:,})" for st in steps]
    hs = [0.25 * (t.count('\n') + 1) + 0.18 for t in mtxts]
    for i, st in enumerate(steps):
        box(xm, ys[i], wm, hs[i], mtxts[i], bold=(i in (0, n - 1)))
        if i == n - 1:
            break
        arrow(xm, ys[i] - hs[i] / 2, xm, ys[i + 1] + hs[i + 1] / 2)
        ex = steps[i + 1]
        nx = ex.get('n_excluded')
        if nx is None or (isinstance(nx, float) and np.isnan(nx)):
            continue
        ymid = (ys[i] - hs[i] / 2 + ys[i + 1] + hs[i + 1] / 2) / 2
        txt = wrap(f"Excluded: {ex['side_label']} (n = {int(nx):,})", 42)
        if ex.get('side_detail'):
            txt += '\n' + wrap(ex['side_detail'], 46)
        lines = txt.count('\n') + 1
        ax.plot([xm, xs - ws / 2], [ymid, ymid], color=INK2, lw=0.9)
        arrow(xs - ws / 2 - 0.15, ymid, xs - ws / 2, ymid)
        box(xs, ymid, ws, 0.24 * lines + 0.22, txt, fc=SURF)
    hm = hs[-1]
    yb = ys[-1] - 1.45
    xb = [xm - 1.45, xm + 1.45]
    ax.plot([xm, xm], [ys[-1] - hm / 2, ys[-1] - 0.75], color=INK2, lw=0.9)
    ax.plot([xb[0], xb[1]], [ys[-1] - 0.75, ys[-1] - 0.75], color=INK2, lw=0.9)
    for x, (lab, k) in zip(xb, final_split):
        arrow(x, ys[-1] - 0.75, x, yb + 0.33)
        box(x, yb, 2.6, 0.66, f"{wrap(lab, 22)}\n(n = {k:,})")
    _save(fig, path)


# ---------------------------------------------------------------------------------------------- CAH x expansion
def cah_expansion_plot(cells, path, ylabel='Adjusted mean uncompensated care ratio'):
    """cells: list of dicts with cah ('CAH'/'Non-CAH'), expansion ('Expansion'/'Non-expansion'), mean, lo, hi."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    groups = ['Non-CAH', 'CAH']
    offs = {'Non-expansion': -0.12, 'Expansion': 0.12}
    cols = {'Non-expansion': ORANGE, 'Expansion': BLUE}
    for e in ['Non-expansion', 'Expansion']:
        xs, ms, los, his = [], [], [], []
        for gi, g in enumerate(groups):
            c = [r for r in cells if r['cah'] == g and r['expansion'] == e][0]
            xs.append(gi + offs[e]); ms.append(c['mean']); los.append(c['lo']); his.append(c['hi'])
        ms, los, his = np.array(ms), np.array(los), np.array(his)
        ax.errorbar(xs, ms, yerr=[ms - los, his - ms], fmt='o', ms=6, color=cols[e], ecolor=cols[e], elinewidth=1.6, capsize=0,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=f'{e} states')
        for x, m in zip(xs, ms):
            ax.text(x + 0.05, m, f'{m:.3f}', va='center', ha='left', fontsize=7.5, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(groups); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    _grid(ax, 'y')
    ax.legend(loc='upper left', fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------------------------- SHAP
def _beeswarm_ax(ax, shap_vals, feat_vals, labels, xlabel, rng=np.random.default_rng(1)):
    k = shap_vals.shape[1]
    for j in range(k):
        y0 = k - 1 - j
        s = shap_vals[:, j]
        v = feat_vals[:, j].astype(float)
        ok = np.isfinite(v)
        col = np.full(len(v), 0.5)
        if ok.sum():
            lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            col[ok] = np.clip((v[ok] - lo) / (hi - lo), 0, 1) if hi > lo else 0.5
        # density-scaled vertical jitter
        bins = np.linspace(s.min(), s.max() + 1e-12, 80)
        idx = np.digitize(s, bins)
        counts = np.bincount(idx, minlength=len(bins) + 1)
        jit = (rng.random(len(s)) - 0.5) * 0.8 * (counts[idx] / counts.max()) ** 0.5
        ax.scatter(s[ok], y0 + jit[ok], c=col[ok], cmap=DIVERGING, vmin=0, vmax=1, s=3.5, linewidths=0, alpha=0.85, rasterized=True)
        if (~ok).any():
            ax.scatter(s[~ok], y0 + jit[~ok], c='#d9d8d4', s=3.5, linewidths=0, alpha=0.85, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(k)); ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_xlabel(xlabel)
    _grid(ax, 'x')


def _colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(['Low', 'High']); cb.set_label('Feature value', fontsize=8); cb.outline.set_visible(False)


def beeswarm(shap_vals, feat_vals, labels, xlabel, path):
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(labels) + 1.2))
    _beeswarm_ax(ax, shap_vals, feat_vals, labels, xlabel)
    _colorbar(fig, ax)
    _save(fig, path)


def _domain_ax(ax, dt, xlabel, color=BLUE):
    dt = dt.sort_values('mean_abs_shap')
    y = np.arange(len(dt))
    ax.barh(y, dt['mean_abs_shap'], height=0.55, color=color)
    ax.hlines(y, dt['q25'], dt['q75'], color=INK2, lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(dt['domain'], fontsize=8)
    ax.set_xlabel(xlabel)
    _grid(ax, 'x')


def domain_bars(dt_clf, dt_reg, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={'wspace': 0.95})
    _domain_ax(axes[0], dt_clf, 'Mean |SHAP|, log-odds of UCR > 0.03')
    axes[0].set_title('A. Classification', loc='left', fontsize=9, fontweight='bold')
    _domain_ax(axes[1], dt_reg, 'Mean |SHAP|, UCR scale')
    axes[1].set_title('B. Regression', loc='left', fontsize=9, fontweight='bold')
    _save(fig, path)


def shap_main_figure(dt_clf, shap_vals, feat_vals, labels, path):
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    _domain_ax(ax1, dt_clf, 'Mean |SHAP| (log-odds); line = interquartile range across model fits')
    ax1.set_title('A. Feature groups', loc='left', fontsize=9, fontweight='bold')
    ax2 = fig.add_subplot(gs[1])
    _beeswarm_ax(ax2, shap_vals, feat_vals, labels, 'SHAP value (log-odds of UCR > 0.03)')
    ax2.set_title('B. Individual features (stable top features)', loc='left', fontsize=9, fontweight='bold')
    _colorbar(fig, ax2)
    _save(fig, path)


def subgroup_domain_bars(sub, path):
    tasks = [('clf', 'Mean |SHAP|, log-odds'), ('reg', 'Mean |SHAP|, UCR scale')]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={'wspace': 0.95})
    for ax, (task, xl) in zip(axes, tasks):
        doms = sorted(set().union(*[set(v['domain']) for (lab, t), v in sub.items() if t == task]))
        order = sub.get(('Non-CAH', task), list(sub.values())[0]).set_index('domain').reindex(doms)['mean_abs_shap'].sort_values().index
        y = np.arange(len(order))
        for k, (lab, col) in enumerate([('CAH', ORANGE), ('Non-CAH', BLUE)]):
            if (lab, task) not in sub:
                continue
            v = sub[(lab, task)].set_index('domain').reindex(order)['mean_abs_shap'].fillna(0)
            ax.barh(y + (0.2 if k == 0 else -0.2), v.values, height=0.36, color=col, label=lab)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.set_xlabel(xl)
        ax.set_title('A. Classification' if task == 'clf' else 'B. Regression', loc='left', fontsize=9, fontweight='bold')
        _grid(ax, 'x')
        ax.legend(loc='lower right', fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------------------------- ROC / calibration
def roc_panel(curves, path, title=None):
    """curves: list of (label, fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, (lab, fpr, tpr, auc) in enumerate(curves):
        ax.plot(fpr, tpr, color=SERIES[i % len(SERIES)], lw=2, label=f'{lab} (AUROC {auc:.2f})')
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, linestyle=(0, (1, 2)))
    ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _grid(ax, 'both')
    ax.legend(loc='lower right', fontsize=7.5)
    if title:
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    _save(fig, path)


def calibration_panel(curves, path):
    """curves: list of (label, mean_pred_by_bin, obs_by_bin)."""
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for i, (lab, x, y) in enumerate(curves):
        ax.plot(x, y, color=SERIES[i % len(SERIES)], lw=2, marker='o', ms=4, markeredgecolor=SURF, label=lab)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, linestyle=(0, (1, 2)))
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed proportion with UCR > 0.03'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _grid(ax, 'both')
    ax.legend(loc='upper left', fontsize=7.5)
    _save(fig, path)


def line_by_cutoff(df, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, (lab, g) in enumerate(df.groupby('model', sort=False)):
        ax.plot(g['cutoff'], g['auroc'], color=SERIES[i], lw=2, marker='o', ms=5, markeredgecolor=SURF, label=lab)
    ax.set_xlabel('Cutoff defining high uncompensated care burden (UCR)'); ax.set_ylabel('AUROC')
    _grid(ax, 'y')
    ax.legend(loc='lower right', fontsize=8)
    _save(fig, path)


def workflow_diagram(path):
    """Figure 1: overview of data sources and analyses."""
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, title, body, fc='#f5f5f3'):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', linewidth=0.9,
                                    edgecolor=INK2, facecolor=fc))
        ax.text(x, y + h / 2 - 0.22, title, ha='center', va='top', fontsize=8.6, fontweight='bold')
        ax.text(x, y + h / 2 - 0.62, body, ha='center', va='top', fontsize=7.3, linespacing=1.35, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))

    box(5, 9.0, 9.4, 1.6, 'Data sources',
        'AHA Annual Survey 2023: structure, organization, social needs activities\n'
        'Medicare cost reports: FY2023 (analysis) and FY2024 (temporal validation)\n'
        'KFF Medicaid expansion status; county data from Census (SAHIE, SAIPE) and BLS (LAUS), 2023')
    box(5, 6.75, 9.4, 1.9, 'Analytic sample and variables',
        'General medical and surgical hospitals, 50 states and DC (n = 4,114; Figure 2)\n'
        'Outcome: uncompensated care ratio (UCR); high burden: UCR > 0.03\n'
        'Nested predictor sets for both methods: A (core), B (+ organization, social needs),\n'
        'C (+ county socioeconomic conditions, payer mix); missing data handled by imputation')
    box(2.55, 3.95, 4.5, 2.5, 'Inferential analysis',
        'Beta regression (logit link)\nstate-clustered standard errors\nadjusted means; CAH x Medicaid expansion\n'
        'models with sets A, B, C; CAH and non-CAH\nsensitivity analyses')
    box(7.45, 3.95, 4.5, 2.5, 'Machine learning',
        'XGBoost, random forest, MLP, elastic net\nbenchmarks: logistic and beta regression\n'
        'repeated nested cross-validation\nregression (UCR) and classification (UCR > 0.03)\nCAH and non-CAH subgroups')
    box(5, 1.0, 9.4, 1.45, 'Interpretation and validation',
        'SHAP values (out of fold): feature and feature-group importance, stability across fits\n'
        'Temporal validation: identification of high burden in FY2024')
    arrow(5, 8.2, 5, 7.72)
    arrow(3.6, 5.8, 2.9, 5.22)
    arrow(6.4, 5.8, 7.1, 5.22)
    arrow(2.55, 2.7, 3.6, 1.75)
    arrow(7.45, 2.7, 6.4, 1.75)
    _save(fig, path)


# ---------------------------------------------------------------------------------------------- main-text ROC figure
NEUTRAL = '#8a8984'


def roc_two_panel(panels, path):
    """panels: list of (title, curves); curves: list of (label, fpr, tpr, auc, color, linestyle)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.1), gridspec_kw={'wspace': 0.28})
    axes = np.atleast_1d(axes)
    for ax, (title, curves) in zip(axes, panels):
        for lab, fpr, tpr, auc, col, ls in curves:
            ax.plot(fpr, tpr, color=col, lw=2, linestyle=ls, label=f'{lab} ({auc:.2f})')
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, linestyle=(0, (1, 2)))
        ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_aspect('equal')
        _grid(ax, 'both')
        leg = ax.legend(loc='lower right', fontsize=7, title='Model (AUROC)', title_fontsize=7, frameon=True, framealpha=0.95)
        leg.get_frame().set_edgecolor('none')
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
    _save(fig, path)


# ---------------------------------------------------------------------------------------------- SHAP comparison figure
SHORT = {'Emergency department visits (log)': 'ED visits (log)', 'Critical access hospital': 'Critical access hospital',
         'County uninsured rate (<65 years)': 'County uninsured rate', 'Medicaid share of inpatient days': 'Medicaid share of days',
         'Self-pay share of discharges': 'Self-pay share', 'County median household income (log)': 'County median income',
         'Medicaid expansion state': 'Medicaid expansion', 'Utilization estimated by AHA': 'Utilization estimated',
         'Payer mix estimated by AHA': 'Payer mix estimated', 'Social needs screening (domains)': 'Social needs screening',
         'Social needs programs (domains)': 'Social needs programs', 'Teaching, accreditation and designations': 'Teaching and designations',
         'Centralized physician/insurance system': 'Physician/insurance system', 'Operates subsidiary corporations': 'Subsidiary corporations',
         'Independent hospital system': 'Independent system', 'Decentralized health system': 'Decentralized system',
         'Centralized health system': 'Centralized system', 'Joint Commission accreditation': 'Joint Commission',
         'Long-term nursing home unit': 'Nursing home unit', 'Health equity strategic planning': 'Equity strategic planning',
         'Health equity accountability': 'Equity accountability', 'Screening coverage (none/some/all)': 'Screening coverage'}


def short(label):
    return SHORT.get(label, label)


def shap_comparison_figure(panels, domains, path, xlabel, dom_xlabel):
    """panels: list of (title, shap_vals, feat_vals, labels) for the 3 populations; domains: list of (name, domain table, color)."""
    fig = plt.figure(figsize=(9.6, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.95)
    axes = []
    for k, (title, sv, fv, labels) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        _beeswarm_ax(ax, sv, fv, [short(x) for x in labels], xlabel)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
        axes.append(ax)
    _colorbar(fig, axes[-1])
    ax = fig.add_subplot(gs[1, :])
    base = domains[0][1].set_index('domain')['mean_abs_shap']
    order = list(base.sort_values().index)
    y = np.arange(len(order))
    h = 0.8 / len(domains)
    for k, (name, dt, col) in enumerate(domains):
        v = dt.set_index('domain').reindex(order)['mean_abs_shap']
        off = (len(domains) - 1) / 2 - k
        ax.barh(y + off * h, v.fillna(0).values, height=h * 0.9, color=col, label=name)
    ax.set_yticks(y); ax.set_yticklabels([short(o) for o in order], fontsize=8)
    ax.set_xlabel(dom_xlabel)
    _grid(ax, 'x')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('D. Feature groups (mean absolute SHAP value of the summed contributions of the features in each group)', loc='left', fontsize=9, fontweight='bold')
    _save(fig, path)
