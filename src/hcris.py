"""Extract hospital financial fields from public CMS HCRIS (Hospital 2552-10) files.

Used for (1) validating the FY2023 extract that the analysis is built on and (2) building the FY2024 outcome
for the temporal validation. HCRIS files (no header rows) are downloaded from
https://downloads.cms.gov/Files/hcris/HOSP10FY{year}.ZIP and unzipped into data_public/hcris/FY{year}/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RPT_COLS = ['rpt_rec_num', 'prvdr_ctrl_type_cd', 'prvdr_num', 'npi', 'rpt_stus_cd', 'fy_bgn_dt', 'fy_end_dt', 'proc_dt',
            'initl_rpt_sw', 'last_rpt_sw', 'trnsmtl_num', 'fi_num', 'adr_vndr_cd', 'fi_creat_dt', 'util_cd', 'npr_dt',
            'spec_ind', 'fi_rcpt_dt']
NMRC_COLS = ['rpt_rec_num', 'wksht_cd', 'line_num', 'clmn_num', 'itm_val_num']

# (worksheet, line, column) -> field name. Line/column codes are 5-digit strings as stored in HCRIS.
FIELDS = {
    ('S100001', '03100', '00100'): 'total_uc_cost_l31',      # total unreimbursed and uncompensated care cost (S-10 Part I line 31; revised S-10 for periods beginning on or after 10/1/2022)
    ('S100001', '03000', '00100'): 'uc_cost_l30',            # cost of uncompensated care: charity care + non-Medicare bad debt (line 30)
    ('G300000', '00100', '00100'): 'total_patient_revenues', # total patient revenues (gross charges; G-3 line 1)
    ('G300000', '00300', '00100'): 'net_patient_revenues',   # net patient revenues (G-3 line 3)
    ('G300000', '00400', '00100'): 'total_operating_expenses',  # total operating expenses (G-3 line 4)
}


def _files(hcris_dir, year):
    d = Path(hcris_dir) / f'FY{year}'
    return d / f'HOSP10_{year}_rpt.csv', d / f'HOSP10_{year}_nmrc.csv'


def extract(hcris_dir, year, chunksize=2_000_000):
    """One row per CCN (prvdr_num) with the FIELDS above. If a provider filed several reports for the fiscal year,
    the report covering the most days is kept (ties: latest period end, then latest processing date)."""
    rpt_path, nmrc_path = _files(hcris_dir, year)
    rpt = pd.read_csv(rpt_path, header=None, names=RPT_COLS, dtype={'prvdr_num': str, 'rpt_rec_num': np.int64})
    for c in ['fy_bgn_dt', 'fy_end_dt', 'proc_dt']:
        rpt[c] = pd.to_datetime(rpt[c], format='%m/%d/%Y', errors='coerce')
    rpt['days'] = (rpt['fy_end_dt'] - rpt['fy_bgn_dt']).dt.days + 1
    keep_ws = {k[0] for k in FIELDS}
    parts = []
    for ch in pd.read_csv(nmrc_path, header=None, names=NMRC_COLS, chunksize=chunksize,
                          dtype={'rpt_rec_num': np.int64, 'wksht_cd': str, 'line_num': str, 'clmn_num': str, 'itm_val_num': float}):
        ch = ch[ch['wksht_cd'].isin(keep_ws)]
        key = list(zip(ch['wksht_cd'], ch['line_num'], ch['clmn_num']))
        ch = ch.assign(field=[FIELDS.get(k) for k in key]).dropna(subset=['field'])
        parts.append(ch[['rpt_rec_num', 'field', 'itm_val_num']])
    vals = pd.concat(parts).pivot_table(index='rpt_rec_num', columns='field', values='itm_val_num', aggfunc='sum')
    out = rpt.merge(vals, left_on='rpt_rec_num', right_index=True, how='left')
    out = (out.sort_values(['prvdr_num', 'days', 'fy_end_dt', 'proc_dt'])
              .drop_duplicates('prvdr_num', keep='last'))
    for f in FIELDS.values():
        if f not in out.columns:
            out[f] = np.nan
        out[f] = out[f].fillna(0.0)   # HCRIS stores only non-zero cells; an absent cell is a zero/blank entry
    return out[['prvdr_num', 'rpt_rec_num', 'fy_bgn_dt', 'fy_end_dt', 'days'] + list(FIELDS.values())].reset_index(drop=True)
