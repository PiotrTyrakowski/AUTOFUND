import json
from typing import Dict, List
from numerai_automl.config.config import MAIN_TARGET, TARGET_CANDIDATES, LIGHTGBM_PARAMS_OPTION, FEATURE_NEUTRALIZATION_PROPORTIONS
import os
import pandas as pd
import cloudpickle
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.utils.utils import get_project_root


target_candidates = TARGET_CANDIDATES
lightgbm_params = LIGHTGBM_PARAMS_OPTION
main_target = MAIN_TARGET
feature_neutralization_proportions = FEATURE_NEUTRALIZATION_PROPORTIONS

class MetaModelManager:
    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 targets_names_for_base_models: List[str] = target_candidates,
                 main_target: str = main_target):
        self.data_version = data_version
        self.feature_set = feature_set
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version=data_version, feature_set=feature_set)
        self.features = self.data_manager.get_features()
        self.base_model_manager = BaseModelManager(
            data_version=data_version,
            feature_set=feature_set,
            targets_names_for_base_models=targets_names_for_base_models
        )
        self.ensemble_model_manager = EnsembleModelManager(
            data_version=data_version,
            feature_set=feature_set,
            targets_names_for_base_models=targets_names_for_base_models
        )
        self.feature_neutralizer = FeatureNeutralizer(all_features=self.features, target_name=main_target)


    def create_neutralized_predictions(self, X: pd.DataFrame):
        X = X[self.features]

        base_models = self.base_model_manager.load_base_models()
        neutralization_params = self.base_model_manager.load_neutralization_params()

        neutralized_predictions = X.copy()

        for target_name in self.targets_names_for_base_models:
            base_model_name = f"model_{target_name}"
            base_model = base_models[base_model_name]

            preditions_name = f"predictions_{base_model_name}"
            neutralized_predictions[preditions_name] = base_model.predict(X)

            neutralized_predictions[f"neutralized_{preditions_name}"] = self.feature_neutralizer.apply_neutralization(neutralized_predictions, preditions_name, neutralization_params[preditions_name]["neutralization_params"])

        return neutralized_predictions



    def create_weighted_meta_model(self, X: pd.DataFrame):
        weighted_ensembler = self.ensemble_model_manager.load_ensemble_model("weighted")
        neutralized_predictions = self.create_neutralized_predictions(X)

        print(neutralized_predictions.columns)

        # print(X.columns)
        # print(weighted_ensembler.best_ensemble_features_and_weights)
        return weighted_ensembler.predict(neutralized_predictions)

        

    def save_meta_model(self):
        pass

    def load_meta_model(self):
        pass