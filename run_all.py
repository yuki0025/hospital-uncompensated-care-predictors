"""Run the analysis of "Identifying Predictors of Hospital Uncompensated Care Burden in the United States: Cross-Sectional
Study Integrating Beta Regression and Interpretable Machine Learning" (JMIR Medical Informatics).

    python run_all.py --data-dir <licensed input files> --out-dir <output folder>                     # all steps
    python run_all.py --data-dir <licensed input files> --out-dir <output folder> --steps ml,report   # selected steps
Steps, in order: sample, missing, measures, inferential, ml, cah, shap, ao7, temporal, report
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import parameters as config
from src import features as F
from src.io_utils import log, out_path, save_json, versions

HERE = Path(__file__).resolve().parent
ALL_STEPS = ['sample', 'missing', 'measures', 'inferential', 'ml', 'cah', 'shap', 'ao7', 'temporal', 'report']
LEARNERS = ['xgb', 'rf', 'mlp', 'en']


def load():
    d = pd.read_parquet(out_path('data/analytic.parquet'))
    return d, F.build_sets(d, config.VARLIST_CSV)


def step_sample():
    from src import sample
    sample.build()


def step_missing():
    from src import missing
    d, S = load()
    res = missing.run_mechanism(d, S)
    save_json(res, 'results/missing_mechanism.json')
    m = config.M_IMPUTATIONS
    imp = missing.impute(S['C'], np.log(d['ucr'] / (1 - d['ucr'])), m=m, cycles=config.MICE_CYCLES, by=d['cah'].values)
    pd.concat([X.assign(_imp=j) for j, X in enumerate(imp)]).to_parquet(out_path('data/imputed_setC.parquet'))
    log(f'   imputed data sets saved: {m}')


def step_measures():
    from src.io_utils import save_df
    d, _ = load()
    mt, imap = F.measures_table(d, config.VARLIST_CSV)
    save_df(mt, 'tables/sdoh_measures.csv')
    save_df(imap, 'tables/sdoh_item_map.csv')
    log('   social needs and health equity measures:', mt[['measure', 'n_items', 'cronbach_alpha']].round(2).to_dict('records'))


def load_imputed(S):
    z = pd.read_parquet(out_path('data/imputed_setC.parquet'))
    imp = [g.drop(columns='_imp').set_index(S['C'].index) for _, g in z.groupby('_imp')]
    return {'C': imp, 'B': [X[S['B'].columns] for X in imp]}


def step_inferential():
    from src import inferential
    d, S = load()
    mi = load_imputed(S)
    zf = pd.read_parquet(out_path('data/frame_zero_incl.parquet'))
    inferential.run(d, S, zf, F.build_sets(zf, config.VARLIST_CSV), mi)
    inferential.delta_adjustment(d, S, mi['B'], m=min(10, len(mi['B'])))


def step_ml():
    from src import ml
    d, S = load()
    ucr, high = d['ucr'].values, d['high_ucr'].values
    folds = ml.outer_folds(ucr, d['cah'].values)
    save_json([[int(r), int(k), te.tolist()] for r, k, _, te in folds], 'data/folds_pooled.json')
    bench = ['lr_cah', 'lr5', 'lr5_exp', 'lr_a', 'beta_a', 'null']
    ml.run_spec('pooled_bench', S['A'], ucr, high, folds, bench)
    ml.run_spec('pooled_C', S['C'], ucr, high, folds, LEARNERS)
    ml.run_spec('pooled_A', S['A'], ucr, high, folds, LEARNERS, shap_learner=None)
    ml.run_spec('pooled_B', S['B'], ucr, high, folds, LEARNERS, shap_learner=None)
    ml.run_spec('pooled_C_noCAH', S['C'].drop(columns=['cah']), ucr, high, folds, ['xgb'], shap_learner=None)
    # CAH and non-CAH subgroups (models trained and evaluated within each subgroup)
    for lab, sel in [('cah', d['cah'].values == 1), ('noncah', d['cah'].values == 0)]:
        idx = np.where(sel)[0]
        f_s = ml.outer_folds(ucr[idx])
        Xs = S['C'].iloc[idx].drop(columns=['cah'])
        ml.run_spec(f'{lab}_C', Xs, ucr[idx], high[idx], f_s, LEARNERS + ['lr5_exp', 'lr_a', 'beta_a', 'null'])
    # Secondary outcome measured at cost (S-10 line 30 / total operating expenses); high = upper quartile
    ok = d['ucr30_opex'].notna().values
    idx = np.where(ok)[0]
    y2 = d['ucr30_opex'].values[idx]
    h2 = (y2 > np.quantile(y2, 0.75)).astype(int)
    f2 = ml.outer_folds(y2, d['cah'].values[idx])
    ml.run_spec('secondary_C', S['C'].iloc[idx], y2, h2, f2, ['xgb', 'en', 'lr5_exp', 'lr_a', 'beta_a', 'null'], shap_learner=None)
    # Robustness of the classification results to the cutoff (first repeat only)
    f0 = [f for f in folds if f[0] == 0]
    for c in [0.02, 0.025, 0.035, 0.04]:
        ml.run_spec(f'cutoff_{c}', S['C'], ucr, (ucr > c).astype(int), f0, ['xgb', 'lr5_exp'], tasks=('clf',), shap_learner=None)


def step_cah():
    from src import cah_analysis
    d, _ = load()
    cah_analysis.run(d)
    cah_analysis.identifier_control(d)


def step_shap():
    from src import shap_analysis
    d, S = load()
    shap_analysis.run(d, S)


def step_ao7():
    from src import ao7
    d, S = load()
    ao7.run(d, S)


def step_temporal():
    from src import temporal
    d, S = load()
    temporal.run(d, S)


def step_report():
    from src import report
    report.run()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='folder with AHA2023.csv, CMS_Cost_Report_2023(Sheet1).csv, and Variable_list_AHA.csv (not distributed)')
    ap.add_argument('--out-dir', required=True, help='folder for all outputs; keep it outside the repository (hospital-level data)')
    ap.add_argument('--hcris-dir', default=str(HERE / 'data_public' / 'hcris'),
                    help='folder with the unzipped HCRIS files in FY2023/ and FY2024/ (default: data_public/hcris)')
    ap.add_argument('--n-jobs', type=int, default=10, help='parallel jobs (does not change the results)')
    ap.add_argument('--steps', default=','.join(ALL_STEPS))
    a = ap.parse_args()
    config.DATA_DIR = Path(a.data_dir)
    config.AHA_CSV = config.DATA_DIR / 'AHA2023.csv'
    config.CMS_CSV = config.DATA_DIR / 'CMS_Cost_Report_2023(Sheet1).csv'
    config.VARLIST_CSV = config.DATA_DIR / 'Variable_list_AHA.csv'
    config.PUBLISHED_IDS_CSV = config.DATA_DIR / 'merged_AHA_data.csv'      # optional; see README
    config.PUBLIC_DIR = HERE / 'data_public'
    config.HCRIS_DIR = Path(a.hcris_dir)
    config.OUT_DIR = Path(a.out_dir)
    config.N_JOBS = a.n_jobs
    save_json({'versions': versions(), 'n_jobs': config.N_JOBS, 'outer_repeats': config.OUTER_REPEATS,
               'n_iter': config.N_ITER, 'm_imputations': config.M_IMPUTATIONS}, 'logs/run_settings.json')
    for s in a.steps.split(','):
        log(f'=== step: {s}')
        globals()[f'step_{s.strip()}']()
    log('=== done')
