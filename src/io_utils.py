"""Logging and output helpers."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src import parameters as config

_T0 = time.time()
_LOG = None


def out_path(rel):
    p = Path(config.OUT_DIR) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(*args):
    global _LOG
    msg = ' '.join(str(a) for a in args)
    line = f'[{time.time() - _T0:7.0f}s] {msg}'
    print(line, flush=True)
    if _LOG is None:
        _LOG = open(out_path('logs/run.log'), 'a')
    _LOG.write(line + '\n')
    _LOG.flush()


def save_df(df, rel, index=False):
    p = out_path(rel)
    df.to_csv(p, index=index)
    return p


def save_json(obj, rel):
    p = out_path(rel)
    p.write_text(json.dumps(obj, indent=1, default=str))
    return p


def versions():
    import numpy, scipy, sklearn, statsmodels, xgboost
    v = {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__,
         'scikit-learn': sklearn.__version__, 'statsmodels': statsmodels.__version__, 'xgboost': xgboost.__version__}
    return v
