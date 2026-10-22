"""Step 1. Link the AHA Annual Survey to the FY2023 cost reports, apply the eligibility rules, derive the outcome and
covariates, and link state Medicaid expansion and county socioeconomic data.

Outputs (OUT_DIR):
  data/analytic.parquet        analytic sample (one row per hospital; contains licensed AHA fields: do not share)
  data/frame_zero_incl.parquet general medical/surgical hospitals incl. those reporting zero uncompensated care (CO2 check)
  tables/flow.csv              exclusion counts for the flow diagram
  tables/zero_uc_profile.csv   who reports zero uncompensated care
  tables/reconciliation.csv    original submission sample vs corrected sample, by state
"""
import numpy as np
import pandas as pd

from src import parameters as config
from src.io_utils import log, out_path, save_df

TERRITORIES = {'Puerto Rico', 'Guam', 'Virgin Islands', 'American Samoa', 'Northern Mariana Islands', 'Marshall Islands',
               'PR', 'GU', 'VI', 'AS', 'MP', 'MH'}
REGIONS = {'Northeast': 'CT ME MA NH RI VT NJ NY PA', 'Midwest': 'IL IN MI OH WI IA KS MN MO NE ND SD',
           'South': 'DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX', 'West': 'AZ CO ID MT NV NM UT WY AK CA HI OR WA'}
STATE_TO_REGION = {s: r for r, ss in REGIONS.items() for s in ss.split()}


def read_inputs():
    # Identifiers are read as strings. With pandas' default chunked type inference the AHA ID column was parsed as a mix of
    # strings and integers, which silently dropped hospitals in the original linkage.
    aha = pd.read_csv(config.AHA_CSV, low_memory=False, dtype={'AHA ID': str, 'Medicare Provider ID': str})
    cms = pd.read_csv(config.CMS_CSV, low_memory=False, dtype={'AHA ID': str, 'Medicare Provider ID': str})
    return aha, cms


def ownership(x):
    x = str(x)
    if x.startswith('Investor-owned'):
        return 'For-profit'
    if x.startswith('Nongovt.'):
        return 'Not-for-profit'
    return 'Government'   # state/local government hospitals (and any federal hospital with a cost report)


def link_external(d):
    """Medicaid expansion (KFF) and county socioeconomic measures (Census SAHIE/SAIPE, BLS LAUS, USDA RUCC)."""
    pub = config.PUBLIC_DIR
    kff = pd.read_csv(pub / 'state_medicaid_expansion.csv', dtype={'state_fips': str})
    exp = kff.set_index('state_abbr')
    d['expansion'] = d['st'].map(exp['expanded_by_2023-01-01']).map({'Y': 1, 'N': 0})
    d['exp_transition_or_wi'] = d['st'].isin(['SD', 'NC', 'WI']).astype(int)
    assert d['expansion'].notna().all(), 'state without expansion status'

    # County FIPS from the AHA Census Bureau state and county codes. Connecticut: Census products from 2022 use 9 planning
    # regions instead of the legacy counties (87 FR 34235); hospitals are assigned to the planning region of their town.
    st = pd.to_numeric(d['Census Bureau state code'], errors='coerce').astype('Int64').astype(str).str.zfill(2)
    co = pd.to_numeric(d['Census Bureau county code'], errors='coerce').astype('Int64').astype(str).str.zfill(3)
    d['fips5_legacy'] = st + co
    ct = pd.read_csv(pub / 'ct_town_planning_region.csv', dtype=str).set_index('town')['planning_region_fips5']
    is_ct = d['st'] == 'CT'
    d['fips5'] = d['fips5_legacy']
    d.loc[is_ct, 'fips5'] = d.loc[is_ct, 'City'].map(ct)
    if d.loc[is_ct, 'fips5'].isna().any():
        log('WARNING: Connecticut towns without a planning-region mapping:', d.loc[is_ct & d['fips5'].isna(), 'City'].tolist())
    sahie = pd.read_csv(pub / 'county_sahie2023.csv', dtype={'fips5': str})[['fips5', 'sahie2023_pctui_u65']]
    saipe = pd.read_csv(pub / 'county_saipe2023.csv', dtype={'fips5': str})[['fips5', 'saipe2023_pov_pct_all', 'saipe2023_mhi']]
    rucc = pd.read_csv(pub / 'county_rucc2023.csv', dtype={'fips5': str})[['fips5', 'rucc2023']]
    laus = pd.read_csv(pub / 'county_laus2023.csv', dtype={'fips5': str})[['fips5', 'laus2023_unemp_rate']]
    d = d.merge(sahie, on='fips5', how='left').merge(saipe, on='fips5', how='left').merge(rucc, on='fips5', how='left')
    d = d.merge(laus.rename(columns={'fips5': 'fips5_legacy'}), on='fips5_legacy', how='left')   # LAUS 2023 keeps legacy CT counties
    for c in ['sahie2023_pctui_u65', 'saipe2023_pov_pct_all', 'saipe2023_mhi', 'rucc2023', 'laus2023_unemp_rate']:
        log(f'   county link {c}: {d[c].notna().mean():.4f}')

    # Hospital payer mix from AHA utilization fields. For hospitals with a nursing-home unit the hospital-unit (acute)
    # values are used; the '(est.)' columns are flags (Reported/Estimated/Expanded), not values.
    num = lambda c: pd.to_numeric(d[c], errors='coerce')
    has_nh = num('Inpatient days (nursing home)') > 0
    mcd, ipd = num('Total Medicaid days'), num('Inpatient Days')
    hu_mcd = num('Hospital unit Medicaid days').where(num('Hospital unit Medicaid days').notna(), mcd - num('Nursing home Medicaid days'))
    sp, dis = num('Total facility self-pay discharges'), num('Total facility inpatient discharges')
    hu_sp = num('Hospital unit self-pay discharges').where(num('Hospital unit self-pay discharges').notna(), sp - num('Nursing home self-pay discharges'))
    d['medicaid_day_share'] = np.where(has_nh, hu_mcd / num('Inpatient days (hospital unit)'), mcd / ipd)
    d['selfpay_discharge_share'] = np.where(has_nh, hu_sp / num('Hospital unit total inpatient discharges'), sp / dis)
    for c in ['medicaid_day_share', 'selfpay_discharge_share']:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        d.loc[(d[c] < 0) | (d[c] > 1), c] = np.nan
    d['payer_mix_estimated'] = ((d['Medicaid Inpatient Days (est.)'] != 'Reported') |
                                (d['Self-pay inpatient discharges (est.)'] != 'Reported')).astype(int)
    d['utilization_estimated'] = (d['Admissions (est.)'] != 'Reported').astype(int)
    return d


def derive(d):
    d = d.copy()
    d['st'] = d['State_cms'].fillna(d['State']) if 'State_cms' in d.columns else d['State']
    d['ucr'] = d['Total uncompensated care cost'] / d['Total patient revenues']
    d['high_ucr'] = (d['ucr'] > config.UCR_CUTOFF).astype(int)
    # Secondary (sensitivity) outcome: charity care + non-Medicare bad debt cost (S-10 line 30) / total operating expenses
    ok30 = (d['Uncompensated care cost'] > 0) & (d['Total operating expenses'] > 0)
    d['ucr30_opex'] = np.where(ok30, d['Uncompensated care cost'] / d['Total operating expenses'], np.nan)
    d.loc[d['ucr30_opex'] >= config.UCR_MAX, 'ucr30_opex'] = np.nan
    d['cah'] = (d['Critical Access Hospital'] == 'Yes').astype(int)
    d['own'] = d['Control / Ownership Type'].map(ownership)
    ctl = d['Control / Ownership Type'].astype(str)
    d['federal'] = (ctl.str.contains(r'federal\)|Department of Defense', regex=True) & ~ctl.str.contains('non federal')).astype(int)
    d['size'] = d['Hospital size']
    d['rural'] = (d['Hospital Location'] == 'Rural').astype(int)
    d['system'] = (d['System member'] == 'Yes').astype(int)   # missing = no System ID = not a system member
    d['teach'] = d['Teaching Status']
    d['region'] = d['st'].map(STATE_TO_REGION)
    d['respondent'] = (d['Response code'] == 'Yes').astype(int)
    d['opex_over_gross'] = d['Total operating expenses'] / d['Total patient revenues']
    return link_external(d)


def build():
    aha, cms = read_inputs()
    log('AHA hospitals:', len(aha), '| unique AHA ID:', aha['AHA ID'].nunique())
    cms_l = cms.dropna(subset=['AHA ID'])
    cms_l = cms_l.rename(columns={c: c + '_cms' for c in cms_l.columns if c in aha.columns and c != 'AHA ID'})
    m = aha.merge(cms_l, on='AHA ID', how='left', validate='one_to_one', indicator=True)
    federal = (m['Control / Ownership Type'].astype(str).str.contains(r'federal\)|Department of Defense', regex=True)
               & ~m['Control / Ownership Type'].astype(str).str.contains('non federal'))
    flow = []

    def step(label, remaining, excluded=None, detail=''):
        flow.append({'step': label, 'n_remaining': int(remaining), 'n_excluded': None if excluded is None else int(excluded), 'detail': detail})
        log(f'{label:<75s} remaining={remaining:>5}  excluded={"" if excluded is None else excluded}  {detail}')

    step('AHA Annual Survey 2023 hospitals', len(m))
    linked = m['_merge'] == 'both'
    step('No FY2023 Medicare cost report', int(linked.sum()), int((~linked).sum()), f'federal {int((federal & ~linked).sum())}')
    m = m[linked].copy()
    zero_any = m['Total uncompensated care cost'] <= 0
    gms_any = m['Primary Service'] == 'General medical and surgical'
    save_df(pd.DataFrame({'group': ['linked hospitals', 'reporting zero or negative uncompensated care',
                                    'of these, not general medical and surgical', 'of these, general medical and surgical'],
                          'n': [len(m), int(zero_any.sum()), int((zero_any & ~gms_any).sum()), int((zero_any & gms_any).sum())]}),
            'tables/zero_uc_by_service.csv')
    terr = m['State'].isin(TERRITORIES)
    step('US territory', int((~terr).sum()), int(terr.sum()))
    m = m[~terr]
    gms = m['Primary Service'] == 'General medical and surgical'
    top = m.loc[~gms, 'Primary Service'].value_counts()
    step('Not a general medical and surgical hospital', int(gms.sum()), int((~gms).sum()),
         '; '.join(f'{k} {v}' for k, v in top.head(6).items()) + f'; other {int(top.iloc[6:].sum())}')
    g = m[gms].copy()
    tuc, tpr = g['Total uncompensated care cost'], g['Total patient revenues']
    no_uc = ~(tuc > 0) | ~(tpr > 0)
    md = g['State'] == 'MD'
    fed_g = federal.loc[g.index]
    step('No uncompensated care or patient revenue reported', int((~no_uc).sum()), int(no_uc.sum()),
         f'zero {int((tuc == 0).sum())} (federal {int((fed_g & (tuc == 0)).sum())}, Maryland {int((md & (tuc == 0)).sum())}); '
         f'negative {int((tuc < 0).sum())}; zero patient revenue {int((tpr <= 0).sum())}')
    zf = g[(tuc >= 0) & (tpr > 0)]
    zero_frame = derive(zf[(zf['Total uncompensated care cost'] / zf['Total patient revenues']) < config.UCR_MAX])
    prof = g[no_uc]
    save_df(pd.DataFrame({'group': ['all', 'federal', 'Maryland', 'CAH', 'survey non-respondent'],
                          'n': [len(prof), int(fed_g[no_uc].sum()), int(md[no_uc].sum()),
                                int((prof['Critical Access Hospital'] == 'Yes').sum()), int((prof['Response code'] == 'No').sum())]}),
            'tables/zero_uc_profile.csv')
    g = g[~no_uc].copy()
    ratio = g['Total uncompensated care cost'] / g['Total patient revenues']
    bad = ratio >= config.UCR_MAX
    step(f'Implausible ratio (>= {config.UCR_MAX})', int((~bad).sum()), int(bad.sum()))
    d = derive(g[~bad])
    step('Analytic sample', len(d), None, f"CAH {int(d.cah.sum())}; non-CAH {int((d.cah == 0).sum())}; "
         f"survey non-respondents {int((d.respondent == 0).sum())}")
    log('   ownership:', d['own'].value_counts().to_dict(), '| federal in sample:', int(d['federal'].sum()))
    log(f"   UCR > {config.UCR_CUTOFF}: {int(d.high_ucr.sum())}/{len(d)} ({d.high_ucr.mean():.1%}); "
        f"{config.UCR_CUTOFF} is the {(d.ucr <= config.UCR_CUTOFF).mean():.1%} percentile; median UCR {d.ucr.median():.4f}")
    save_df(pd.DataFrame(flow), 'tables/flow.csv')
    d.to_parquet(out_path('data/analytic.parquet'))
    zero_frame.to_parquet(out_path('data/frame_zero_incl.parquet'))
    reconcile(d)
    return d


def reconcile(d):
    """Hospitals in the original submission's analytic file vs the corrected sample, by state."""
    try:
        old = set(pd.read_csv(config.PUBLISHED_IDS_CSV, usecols=['AHA ID'], dtype={'AHA ID': str})['AHA ID'])
    except FileNotFoundError:
        log('original analytic file not found; reconciliation skipped')
        return
    t = d.assign(in_original=d['AHA ID'].isin(old)).groupby('st').agg(n_corrected=('AHA ID', 'size'), n_in_original=('in_original', 'sum'))
    save_df(t.reset_index(), 'tables/reconciliation_by_state.csv')
    log(f'   original analytic file: {len(old)} hospitals; {int(d["AHA ID"].isin(old).sum())} are in the corrected sample; '
        f'states with no hospital in the original file: {int((t.n_in_original == 0).sum())}')


def frame_for_ids(ids):
    """Linked and derived frame (same codings as the analytic sample) for an arbitrary set of AHA IDs; used to rebuild the
    original submission's sample for the replication in the AO7 diagnostic."""
    aha, cms = read_inputs()
    cms_l = cms.dropna(subset=['AHA ID'])
    cms_l = cms_l.rename(columns={c: c + '_cms' for c in cms_l.columns if c in aha.columns and c != 'AHA ID'})
    m = aha.merge(cms_l, on='AHA ID', how='inner', validate='one_to_one')
    return derive(m[m['AHA ID'].isin(set(ids))])
