import json
from typing import Dict, List
from numerai_automl.config.config import MAIN_TARGET, TARGET_CANDIDATES, LIGHTGBM_PARAMS_OPTION, FEATURE_NEUTRALIZATION_PROPORTIONS
import os
import pandas as pd
import cloudpickle
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.utils.utils import get_project_root


target_candidates = TARGET_CANDIDATES
lightgbm_params = LIGHTGBM_PARAMS_OPTION
main_target = MAIN_TARGET
feature_neutralization_proportions = FEATURE_NEUTRALIZATION_PROPORTIONS

class MetaModelManager:
    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 targets_names_for_base_models: List[str] = target_candidates):
        self.data_version = data_version
        self.feature_set = feature_set
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version=data_version, feature_set=feature_set)
       
    def create_weighted_meta_model(self, ):

        pass

    def save_meta_model(self):
        pass

    def load_meta_model(self):
        pass