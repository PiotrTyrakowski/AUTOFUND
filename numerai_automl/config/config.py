# numerai_automl/config.py

DATA_VERSION = "v5.0"
MAIN_TARGET = "target_cyrusd_20"

LIGHTGBM_PARAMS_OPTION = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 15,  # 2**4 - 1
    "colsample_bytree": 0.1,
    "n_jobs": -1
}

# for better accuracy
# LIGHTGBM_PARAMS_OPTION = {
#     "n_estimators": 30000,
#     "learning_rate": 0.001,
#     "max_depth": 10,
#     "num_leaves": 1023,  # 2**10 - 1
#     "colsample_bytree": 0.1,
#     "min_data_in_leaf": 10000
# }

LIGHTGBM_PARAM_GRID={
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'n_estimators': [20, 50, 100, 300, 500, 1000, 1500, 2000],
    'max_depth': [-1, 3, 5, 7, 10, 12],
    'num_leaves': [20, 31, 50, 70, 100, 150],
    'min_child_samples': [10, 20, 50, 100, 200],
    'subsample': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'n_jobs': [-1]
}

ELASTIC_NET_PARAM_GRID = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],  # Regularization strength
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Mixing ratio (L1 vs L2)
}

TARGET_CANDIDATES = [
    "target_agnes_20", "target_agnes_60",
    "target_alpha_20", "target_alpha_60", 
    "target_bravo_20", "target_bravo_60",
    "target_caroline_20", "target_caroline_60",
    "target_charlie_20", "target_charlie_60", 
    "target_claudia_20", "target_claudia_60",
    "target", "target_cyrusd_60", # here is the main target (20)
    "target_delta_20", "target_delta_60",
    "target_echo_20", "target_echo_60",
    "target_jeremy_20", "target_jeremy_60",
    "target_ralph_20", "target_ralph_60", 
    "target_rowan_20", "target_rowan_60",
    "target_sam_20", "target_sam_60",
    "target_teager2b_20", "target_teager2b_60",
    "target_tyler_20", "target_tyler_60",
    "target_victor_20", "target_victor_60",
    "target_waldo_20", "target_waldo_60",
    "target_xerxes_20", "target_xerxes_60"
]

MAIN_TARGET = "target"

FEATURE_NEUTRALIZATION_PROPORTIONS = [0.25, 0.5, 0.75, 1.0]

FEATURE_SET_OPTIONS = ["small", "medium", "large"]
FEATURE_SET_OPTION = "small"
