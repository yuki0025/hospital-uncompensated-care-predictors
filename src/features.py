"""Step 2. Prespecified feature sets shared by the inferential and machine learning analyses.

Sets are nested and identical across methods:
  A  core structural characteristics and Medicaid expansion (complete for every hospital)
  B  A + additional structural/organizational characteristics + social needs (SDOH) activity domains
  C  B + county socioeconomic measures + hospital payer mix
No identifier, fine-grained geography, survey-process field, or cost-report financial field may enter any set: assert_clean()
raises an error if one does.
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------- exclusion lists
ID_GEO_EXACT = {
    'AHA ID', 'Medicare Provider ID', 'Medicare Provider ID_cms', 'CCN', 'MCRNUM', 'SYSID', 'System ID', 'System Name',
    'Hospital Name', 'Address', 'City', 'State', 'State_cms', 'st', 'Zip', 'County', 'fips5', 'fips5_legacy',
    'Latitude', 'Longitude', 'Report number', 'Year',
    # survey-process markers (not hospital characteristics)
    'Response code', 'respondent', 'Days open during reporting period', 'Open 12 full months before end of reporting period',
    'Cost report status', 'Beginning date', 'Ending date',
}
ID_GEO_REGEX = [r'provider id', r'^system (id|name|address|city|state|zip|primary contact|contact title)', r'^hospital name',
                r'^address', r'^zip', r'fips', r'cbsa (code|name)', r'dartmouth', r'latitude', r'longitude', r'telephone',
                r'^area code', r'^county', r'^census bureau', r'statistical area', r'metropolitan division',
                r'^(gpo|distributors|management organization|host hospital) ', r'chief admin', r'_cms$']
LEAK_REGEX = [r'uncompensated', r'bad debt', r'charity', r'expense', r'revenue', r'(total|other|net) income', r'margin',
              r'cost to charge', r'contractual', r'uncollectible', r'medicaid (charges|cost)', r'gross medicaid', r'indigent',
              r'schip', r'payroll', r'assets', r'liabilit', r'fund balance', r'receivable', r'^ucr', r'high_ucr',
              r'opex', r'disproportionate']


def forbidden(col):
    if col in ID_GEO_EXACT:
        return 'identifier, geography, or survey-process field'
    for p in ID_GEO_REGEX:
        if re.search(p, col, flags=re.I):
            return 'identifier or fine-grained geography'
    for p in LEAK_REGEX:
        if re.search(p, col, flags=re.I):
            return 'cost-report financial field (outcome leakage)'
    return None


def assert_clean(X):
    bad = {c: forbidden(c) for c in X.columns if forbidden(c)}
    if bad:
        raise ValueError(f'forbidden features: {bad}')
    return True


YN = {'Yes': 1.0, 'No': 0.0}

# ---------------------------------------------------------------------------------------------- Set A
def set_a(d):
    X = pd.DataFrame(index=d.index)
    X['cah'] = d['cah'].astype(float)
    X['own_forprofit'] = (d['own'] == 'For-profit').astype(float)          # reference: not-for-profit
    X['own_government'] = (d['own'] == 'Government').astype(float)
    X['size_medium'] = (d['size'] == 'Medium').astype(float)                # reference: small (<100 beds)
    X['size_large'] = (d['size'] == 'Large').astype(float)
    X['rural'] = d['rural'].astype(float)
    X['system'] = d['system'].astype(float)
    X['teach_minor'] = (d['teach'] == 'Minor').astype(float)               # reference: non-teaching
    X['teach_major'] = (d['teach'] == 'Major').astype(float)
    for r in ['Northeast', 'Midwest', 'West']:                              # reference: South
        X['region_' + r.lower()] = (d['region'] == r).astype(float)
    X['expansion'] = d['expansion'].astype(float)
    return X


# ---------------------------------------------------------------------------------------------- Set B additions
# AAMC membership duplicates Teaching Status 'Major', and medical-school affiliation and ACGME participation are encoded by
# Teaching Status, so they are represented by teaching status rather than entered separately.
STRUCT_YN = [
    ('Accreditation by The Joint Commission', 'acc_joint_commission'), ('DNV Healthcare Accreditation', 'acc_dnv'),
    ('Accreditation by Accreditation Commission for Healthcare, Inc (ACHC)', 'acc_achc'),
    ('Accreditation by Commission on Accreditation of Rehabilitation Facilities (CARF)', 'acc_carf'),
    ('Cancer program approved by American College of Surgeons', 'acs_cancer_program'), ('Catholic Church Operated', 'catholic'),
    ('Sole Community Provider', 'sole_community_provider'), ('Rural Referral Center', 'rural_referral_center'),
    ('Separate long-term nursing home', 'nursing_home_unit'),
    ('Contract managed hospital', 'contract_managed'), ('Operates subsidiary corps.', 'subsidiary_corps'),
    ('Frontline facility', 'frontline_facility'),
    ('Hospital partnership - certified community behavioral health center', 'partner_ccbhc'),
    ('Hospital partnership - community mental health center', 'partner_cmhc'),
]
CENTRAL = {'Centralized Health System': 'central_centralized', 'Decentralized Health System': 'central_decentralized',
           'Independent Hospital System': 'central_independent', 'Centralized Physician/Insurance Health System': 'central_phys_ins'}
SCREEN_GATE = 'Does your hospital or system screen patients for social needs?'


def structural_extra(d):
    X = pd.DataFrame(index=d.index)
    for c, nm in STRUCT_YN:
        X[nm] = d[c].map(YN).astype(float)
    cen = d['Degree of centralization of health system']
    for lvl, nm in CENTRAL.items():   # reference among system members: moderately centralized; non-members: all 0
        X[nm] = (cen == lvl).astype(float)
    X['cbsa_micro'] = (d['CBSA type'] == 'Micro').astype(float)
    X['log_er_visits'] = np.log1p(pd.to_numeric(d['Emergency room visits'], errors='coerce').astype(float))
    X['log_outpatient_visits'] = np.log1p(pd.to_numeric(d['Total outpatient visits'], errors='coerce').astype(float))
    X['aha_estimated_utilization'] = d['utilization_estimated'].astype(float)
    return X


def sdoh_items(var_list_csv):
    vl = pd.read_csv(var_list_csv)
    items = vl.loc[vl.Category == 'Social Needs and Community Social Determinants of Health', 'VariableName'].tolist()
    return [c for c in items if 'description' not in c]


def item_features(d, items, missing_rule='nan'):
    """SDOH items as 0/1 (screening level 0/1/2). When a hospital does not screen for social needs, the domain-specific
    screening items and the EHR item do not apply and are set to 0. missing_rule='no' reproduces the original coding
    (every missing answer = 'No'); it is used only for the comparison in the missing-data appendix."""
    cols = {}
    for c in items:
        if c == SCREEN_GATE:
            cols['screen_level'] = d[c].map({'No': 0.0, 'Yes for some patients': 1.0, 'Yes for all patients': 2.0}).astype(float)
        else:
            cols[c] = d[c].map(YN).astype(float)
    X = pd.DataFrame(cols, index=d.index)
    no_screen = (d[SCREEN_GATE] == 'No').values
    dom = [c for c in items if c.endswith(' screening')] + ['Does hospital or system record social needs screening results in EHR?']
    dom = [c for c in dom if c in X.columns]
    X.loc[no_screen, dom] = X.loc[no_screen, dom].fillna(0.0)
    if missing_rule == 'no':
        X = X.fillna(0.0)
    return X


def sdoh_domains(cols):
    return {
        'sdoh_screening': [c for c in cols if c.endswith(' screening')],
        'sdoh_programs': [c for c in cols if c.endswith('program/strategy')],
        'sdoh_partnerships': [c for c in cols if re.search(r' - (CHNA|community[- ]level initiatives|social needs)$', c)],
        'sdoh_equity_accountability': [c for c in cols if c.startswith('Accountable for') and 'none of the above' not in c],
        'sdoh_equity_planning': [c for c in cols if c.startswith('Health equity strategic planning') and 'none of the above' not in c],
        'sdoh_data_use': [c for c in cols if (c.startswith('DEI disaggregated') and 'none of the above' not in c)
                          or c.startswith('Does hospital or system record') or c.startswith('Utilization of outcome metrics')],
    }


PERCEIVED_BENEFITS = ('Better health outcomes for patients', 'Decreased health care costs',
                      'Decreased utilization of hospital or system services', 'Improved community health status')
MEASURE_ORDER = ['sdoh_screening', 'sdoh_programs', 'sdoh_partnerships', 'sdoh_equity_accountability', 'sdoh_equity_planning',
                 'sdoh_data_use', 'sdoh_screen_level']


def sdoh_item_map(items):
    """Measure that each item of the AHA social needs and health equity section enters, or why it is not used."""
    where = {c: k for k, cs in sdoh_domains(items).items() for c in cs}
    rows = []
    for c in items:
        if c in where:
            rows.append((c, where[c], ''))
        elif c == SCREEN_GATE:
            rows.append((c, 'sdoh_screen_level', ''))
        elif c in PERCEIVED_BENEFITS:
            rows.append((c, '', 'perceived benefit rather than an activity'))
        elif c.endswith('none of the above') or c.endswith('not involved'):
            rows.append((c, '', 'complement of the other answer options of the question'))
        else:
            rows.append((c, '', 'other'))
    return pd.DataFrame(rows, columns=['item', 'measure', 'not_used_reason'])


def cronbach_alpha(X):
    X = X.dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return np.nan, len(X)
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1)), len(X)


def measures_table(d, var_list_csv):
    """Number of items and internal consistency (Cronbach alpha of the observed answers, among hospitals that answered every item of
    the domain) of the 7 social needs and health equity measures, and the item map."""
    items = sdoh_items(var_list_csv)
    imap = sdoh_item_map(items)
    raw = pd.DataFrame({c: d[c].map(YN) for c in items if c != SCREEN_GATE}, index=d.index)
    rows = []
    for k in MEASURE_ORDER:
        cs = imap.loc[imap.measure == k, 'item'].tolist()
        if k == 'sdoh_screen_level':
            a, n = np.nan, int(d[SCREEN_GATE].notna().sum())
        else:
            a, n = cronbach_alpha(raw[cs])
        rows.append({'measure': k, 'label': LABELS.get(k, k), 'n_items': len(cs), 'cronbach_alpha': a, 'n_answered_all_items': n})
    return pd.DataFrame(rows), imap


def composites(Xi, min_answered=0.8):
    """Domain score = share of the domain's activities the hospital reports; missing if < 80% of its items were answered."""
    C = pd.DataFrame(index=Xi.index)
    for k, cs in sdoh_domains(Xi.columns).items():
        sub = Xi[cs]
        C[k] = sub.mean(axis=1).where(sub.notna().mean(axis=1) >= min_answered)
    C['sdoh_screen_level'] = Xi['screen_level']
    return C


# ---------------------------------------------------------------------------------------------- Set C additions
def ao5_extra(d):
    X = pd.DataFrame(index=d.index)
    X['cty_poverty_pct'] = d['saipe2023_pov_pct_all'].astype(float)
    X['cty_log_median_income'] = np.log(d['saipe2023_mhi'].astype(float))
    X['cty_uninsured_pct'] = d['sahie2023_pctui_u65'].astype(float)
    X['cty_unemployment_pct'] = d['laus2023_unemp_rate'].astype(float)
    X['pm_medicaid_day_share'] = d['medicaid_day_share'].astype(float)
    X['pm_selfpay_discharge_share'] = d['selfpay_discharge_share'].astype(float)
    X['pm_aha_estimated'] = d['payer_mix_estimated'].astype(float)
    return X


def build_sets(d, var_list_csv, missing_rule='nan'):
    items = sdoh_items(var_list_csv)
    A = set_a(d)
    Xi = item_features(d, items, missing_rule=missing_rule)
    B = pd.concat([A, structural_extra(d), composites(Xi)], axis=1)
    C = pd.concat([B, ao5_extra(d)], axis=1)
    for X in (A, B, C):
        assert_clean(X)
    return {'A': A, 'B': B, 'C': C, 'items': Xi}


# ---------------------------------------------------------------------------------------------- labels and domains
LABELS = {
    'cah': 'Critical access hospital', 'own_forprofit': 'For-profit ownership', 'own_government': 'Government ownership',
    'size_medium': 'Medium size (100-399 beds)', 'size_large': 'Large size (>=400 beds)', 'rural': 'Rural location',
    'system': 'System member', 'teach_minor': 'Minor teaching', 'teach_major': 'Major teaching', 'region_northeast': 'Northeast',
    'region_midwest': 'Midwest', 'region_west': 'West', 'expansion': 'Medicaid expansion state',
    'acc_joint_commission': 'Joint Commission accreditation', 'acc_dnv': 'DNV accreditation', 'acc_achc': 'ACHC accreditation',
    'acc_carf': 'CARF accreditation', 'acs_cancer_program': 'ACS-approved cancer program', 'catholic': 'Catholic church operated',
    'sole_community_provider': 'Sole community provider', 'rural_referral_center': 'Rural referral center',
    'nursing_home_unit': 'Long-term nursing home unit', 'contract_managed': 'Contract managed', 'subsidiary_corps': 'Operates subsidiary corporations',
    'frontline_facility': 'Frontline facility', 'partner_ccbhc': 'Partnership with CCBHC', 'partner_cmhc': 'Partnership with CMHC',
    'central_centralized': 'Centralized health system', 'central_decentralized': 'Decentralized health system',
    'central_independent': 'Independent hospital system', 'central_phys_ins': 'Centralized physician/insurance system',
    'cbsa_micro': 'Micropolitan area', 'log_er_visits': 'Emergency department visits (log)', 'log_outpatient_visits': 'Outpatient visits (log)',
    'aha_estimated_utilization': 'Utilization estimated by AHA', 'sdoh_screening': 'Social needs screening (domains)',
    'sdoh_programs': 'Social needs programs (domains)', 'sdoh_partnerships': 'Community partnerships', 'sdoh_equity_accountability': 'Health equity accountability',
    'sdoh_equity_planning': 'Health equity strategic planning', 'sdoh_data_use': 'Social needs data use', 'sdoh_screen_level': 'Screening coverage (none/some/all)',
    'cty_poverty_pct': 'County poverty rate', 'cty_log_median_income': 'County median household income (log)',
    'cty_uninsured_pct': 'County uninsured rate (<65 years)', 'cty_unemployment_pct': 'County unemployment rate',
    'pm_medicaid_day_share': 'Medicaid share of inpatient days', 'pm_selfpay_discharge_share': 'Self-pay share of discharges',
    'pm_aha_estimated': 'Payer mix estimated by AHA',
}
DOMAIN = {}
for f in ['cah']:
    DOMAIN[f] = 'Critical access designation'
for f in ['own_forprofit', 'own_government']:
    DOMAIN[f] = 'Ownership'
for f in ['size_medium', 'size_large', 'log_er_visits', 'log_outpatient_visits']:
    DOMAIN[f] = 'Size and volume'
for f in ['teach_minor', 'teach_major', 'acc_joint_commission', 'acc_dnv', 'acc_achc', 'acc_carf', 'acs_cancer_program', 'nursing_home_unit',
          'sole_community_provider', 'rural_referral_center', 'frontline_facility', 'partner_ccbhc', 'partner_cmhc']:
    DOMAIN[f] = 'Teaching, accreditation and designations'
for f in ['system', 'central_centralized', 'central_decentralized', 'central_independent', 'central_phys_ins', 'contract_managed',
          'subsidiary_corps', 'catholic']:
    DOMAIN[f] = 'System and governance'
for f in ['region_northeast', 'region_midwest', 'region_west', 'expansion', 'rural', 'cbsa_micro']:
    DOMAIN[f] = 'Geography and Medicaid policy'
for f in ['sdoh_screening', 'sdoh_programs', 'sdoh_partnerships', 'sdoh_equity_accountability', 'sdoh_equity_planning', 'sdoh_data_use',
          'sdoh_screen_level']:
    DOMAIN[f] = 'Social needs and health equity activities'
for f in ['cty_poverty_pct', 'cty_log_median_income', 'cty_uninsured_pct', 'cty_unemployment_pct']:
    DOMAIN[f] = 'County socioeconomic conditions'
for f in ['pm_medicaid_day_share', 'pm_selfpay_discharge_share']:
    DOMAIN[f] = 'Payer mix'
for f in ['aha_estimated_utilization', 'pm_aha_estimated']:
    DOMAIN[f] = 'Data provenance flags'
