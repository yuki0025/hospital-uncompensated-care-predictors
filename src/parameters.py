"""Analysis parameters, as reported in the Methods of the manuscript."""

SEED = 20261017                 # random seed for the cross-validation folds, hyperparameter searches, and models
UCR_MAX = 0.5                   # ratios of 0.5 or higher are treated as implausible and excluded
UCR_CUTOFF = 0.03               # high uncompensated care burden: UCR >0.03 (upper quartile)

# Repeated nested cross-validation: 5 outer folds repeated 3 times (15 test folds), 5 inner folds for the search
OUTER_REPEATS = 3
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_ITER = {'xgb': 40, 'rf': 20, 'mlp': 20, 'en': 20}   # randomized-search draws per learner

# Multiple imputation by chained equations
M_IMPUTATIONS = 60
MICE_CYCLES = 20

# Input and output locations and the number of parallel jobs; set by run_all.py from its command-line arguments
DATA_DIR = AHA_CSV = CMS_CSV = VARLIST_CSV = PUBLISHED_IDS_CSV = PUBLIC_DIR = HCRIS_DIR = OUT_DIR = None
N_JOBS = 1
