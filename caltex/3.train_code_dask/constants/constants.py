DEPENDENCIES = ["lightgbm"]
# It is used while launching the job to decide which all dependency packages
# are to be uploaded to the training container.

BINARY_CLASSIFICATION = "binary classification"
MULTI_CLASSIFICATION = "multi-class classification"

# Hyperprameters constants
DEFAULT_NUM_BOOST_ROUND = 20
DEFAULT_EARLY_STOPPING_ROUNDS = 5
DEFAULT_METRIC = "auto"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_LEAVES = 31
DEFAULT_FEATURE_FRACTION = 0.9
DEFAULT_BAGGING_FRACTION = 0.9
DEFAULT_BAGGING_FREQUENCE = 5
DEFAULT_MAX_DEPTH = 12
DEFAULT_MIN_DATA_IN_LEAF = 26
DEFAULT_MAX_DELTA_STEP = 0.0
DEFAULT_LAMBDA_L1 = 0.0
DEFAULT_LAMBDA_L2 = 0.0
DEFAULT_BOOSTING = "gbdt"
DEFAULT_MIN_GAIN_TO_SPLIT = 0.0
DEFAULT_SCALE_POS_WEIGHT = 1.0
DEFAULT_TREE_LEARNER = "serial"
DEFAULT_FEATURE_FRACTION_BYNODE = 1.0
DEFAULT_IS_UNBALANCE = "False"
DEFAULT_MAX_BIN = 255
DEFAULT_NUM_THREADS = 0
DEFAULT_VERBOSITY = 1
DEFAULT_USE_DASK = "False"

AUTO = "auto"

SCHEDULER_IP_TEMPLATE = "tcp://{ip}:8786"
SCHEDULER_PORT = 8786

DASK_WAITTIME_SECONDS = 2

# Problem type - objective mapping
CLASSIFICATION_PROBLEM_TYPE_OBJECTIVE_MAPPING = {
    "binary classification": "binary",
    "multi-class classification": "multiclass",
}

# Problem type - evaluation metric mapping
CLASSIFICATION_PROBLEM_TYPE_METRIC_MAPPING = {
    "binary classification": "binary_logloss",
    "multi-class classification": "multi_logloss",
}

INPUT_MODEL_UNTARRED_PATH = "_input_model_extracted/"
