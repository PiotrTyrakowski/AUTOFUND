import json
from typing import Dict, List
from numerai_automl.config.config import MAIN_TARGET, TARGET_CANDIDATES, LIGHTGBM_PARAMS_OPTION, FEATURE_NEUTRALIZATION_PROPORTIONS
import os
import pandas as pd
import cloudpickle
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.feature_neutralizer.feature_neutralizer import FeatureNeutralizer
from numerai_automl.model_trainer.model_trainer import ModelTrainer
from numerai_automl.utils.utils import get_project_root

# find folder models in my project

target_candidates = TARGET_CANDIDATES
lightgbm_params = LIGHTGBM_PARAMS_OPTION
main_target = MAIN_TARGET
feature_neutralization_proportions = FEATURE_NEUTRALIZATION_PROPORTIONS


class ModelManager():
    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 lightgbm_params: Dict = lightgbm_params,
                 targets_names_for_base_models: List[str] = target_candidates,
                ):
        
        self.project_root = get_project_root()
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version, feature_set)
        self.lightgbm_params = lightgbm_params
        self.base_models = {}
        self.neutralization_params = {}

    def _get_features_names(self, data: pd.DataFrame):
        return [col for col in data.columns if 'feature' in col]
    
    def _get_targets_names(self, data: pd.DataFrame):
        return [col for col in data.columns if 'target' in col]
    
    def _get_predictions_names(self, data: pd.DataFrame):
        return [col for col in data.columns if 'predictions' in col]

    def train_base_models(self):

        train_data = self.data_manager.load_train_data_for_base_models()

        features_names = self._get_features_names(train_data)
        targets_names = self._get_targets_names(train_data)

        assert all(target in targets_names for target in self.targets_names_for_base_models), "Target names are not in targets columns"

        self.base_models = {}
        for target in self.targets_names_for_base_models:
            modelTrainer = ModelTrainer(self.lightgbm_params)
            modelTrainer.train(train_data[features_names], train_data[target])
            model = modelTrainer.get_model()
            self.base_models[f"model_{target}"] = model

    def save_base_models(self):
        if self.base_models is None:
            raise Exception("Models do not exist")
        
        for target, model in self.base_models.items():
            model_path = f"{self.project_root}/models/base_models/model_{target}.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)


    def load_base_models(self):
        self.base_models = {}
        for target in self.targets_names_for_base_models:
            model_path = f"{self.project_root}/models/base_models/model_{target}.pkl"
            with open(model_path, "rb") as f:
                self.base_models[f"model_{target}"] = cloudpickle.load(f)

    def get_base_models(self):
        return self.base_models
    
    
    def create_predictions_for_base_models(self):
        data_for_creating_predictions = self.data_manager.load_data_for_creating_predictions_for_base_models()

        features_names = self._get_features_names(data_for_creating_predictions)

        for target in self.targets_names_for_base_models:
            predictions = self.base_models[f"model_{target}"].predict(data_for_creating_predictions[features_names])
            data_for_creating_predictions[f"predictions_model_{target}"] = predictions

        self.data_manager.save_predictions_for_base_models(data_for_creating_predictions)

        return data_for_creating_predictions
    

    def find_neutralization_features_and_proportions_for_base_models(
            self,
            metric: str = "mean",
            target_name: str = main_target, 
            iterations: int=10, 
            max_number_of_features_to_neutralize: int=6, 
            proportions: List[float]=feature_neutralization_proportions
            ):
        
        if self.base_models is None:
            raise Exception("Base models do not exist")
        

        
        validation_data = self.data_manager.load_validation_data_for_neutralization_of_base_models()

        features_names = self._get_features_names(validation_data)

        predictions_names = self._get_predictions_names(validation_data)

        neutralizer = FeatureNeutralizer(features_names, target_name, iterations, max_number_of_features_to_neutralize, proportions)

        neutralization_params = {}
        for predictions_name in predictions_names:
            neutralization_params[predictions_name] = neutralizer.find_neutralization_features_and_proportions(validation_data[["era", target_name] + features_names], validation_data[predictions_name], metric)

        self.neutralization_params = neutralization_params

        self.save_neutralization_params()
        return neutralization_params
    
    def get_neutralization_params(self):
        return self.neutralization_params
    
    def save_neutralization_params(self):
        if self.neutralization_params is None:
            raise Exception("Neutralization params do not exist")

        with open(f"{self.project_root}/models/neutralization_params/neutralization_params.json", "w") as f:
            json.dump(self.neutralization_params, f, indent=4)
    
    def load_neutralization_params(self):
        with open(f"{self.project_root}/models/neutralization_params/neutralization_params.json", "r") as f:
            self.neutralization_params = json.load(f)
    
           
    def create_neutralized_predictions_from_base_models_predictions(self):

        if self.neutralization_params is None:
            raise Exception("Neutralization params do not exist")
        
        vanila_predictions_data = self.data_manager.load_vanila_predictions_data()
        features_names = self._get_features_names(vanila_predictions_data)
        predictions_names = self._get_predictions_names(vanila_predictions_data)

        neutralized_predictions = vanila_predictions_data.copy()
        neutralizer = FeatureNeutralizer(features_names)

        for target in self.targets_names_for_base_models:
            neutralized_predictions[f"neutralized_predictions_model_{target}"] = neutralizer.apply_neutralization(vanila_predictions_data, f"predictions_model_{target}", self.neutralization_params[f"predictions_model_{target}"]["neutralization_params"])[f"neutralized_predictions_model_{target}"]

        # drop prediction columns
        neutralized_predictions.drop(columns=predictions_names, inplace=True)
        self.data_manager.save_neutralized_predictions_for_base_models(neutralized_predictions)

        return neutralized_predictions


