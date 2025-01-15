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


class EnsembleModelManager:
    def __init__(self, data_version: str = "v5.0", 
                 feature_set: str = "small", 
                 targets_names_for_base_models: List[str] = target_candidates):
        """
        Initialize the ModelManager.

        Args:
            data_version (str): Version of the dataset to use
            feature_set (str): Size of feature set ('small' or 'medium')
            targets_names_for_base_models (List[str]): List of targets to train models for
        """
        self.project_root = get_project_root()
        self.targets_names_for_base_models = targets_names_for_base_models
        self.data_manager = DataManager(data_version, feature_set)




    def find_weighted_ensemble(
            self,
            metric: str = "mean",
            target_name: str = main_target, 
            number_of_iterations: int = 10, 
            max_number_of_prediction_features_for_ensemble: int = 5, 
            number_of_diffrent_weights_for_ensemble: int = 5
            ) -> Dict:
        """
        Find the best weighted ensemble for the neutralized predictions.
        """

        all_neutralized_prediction_features = [f"neutralized_predictions_model_{target_name}" for target_name in self.targets_names_for_base_models]

        train_data = self.data_manager.load_train_data_for_ensembler()
        
        weighted_ensembler = WeightedTargetEnsembler(
            all_neutralized_prediction_features=all_neutralized_prediction_features,
            target_name=target_name,
            number_of_interations=number_of_iterations,
            max_number_of_prediction_features_for_ensemble=max_number_of_prediction_features_for_ensemble,
            number_of_diffrent_weights_for_ensemble=number_of_diffrent_weights_for_ensemble
        )

        self.weighted_ensembler_params = weighted_ensembler.find_ensemble_prediction_features_and_proportions(train_data, metric)
        self.save_weighted_ensembler_params()
        return weighted_ensembler
    
    def get_weighted_ensembler_params(self) -> Dict:
        """
        Retrieve the current weighted ensembler params.
        """
        # assert that the params are not empty
        assert self.weighted_ensembler_params is not None, "Weighted ensembler params are not found"

        return self.weighted_ensembler_params
    
    def save_weighted_ensembler_params(self) -> None:
        """
        Save weighted ensembler params to disk in JSON format.
        Parameters are saved in the project's models/ensemble_models/weighted_ensembler directory.
        """

        # assert that the params are not empty
        assert self.weighted_ensembler_params is not None, "Weighted ensembler params are not found"

        with open(f"{self.project_root}/models/ensemble_models/weighted_ensembler/weighted_ensembler_params.json", "w") as f:
            json.dump(self.weighted_ensembler_params, f, indent=4)

    def load_weighted_ensembler_params(self) -> None:
        """
        Load previously saved weighted ensembler params from disk.
        Parameters are loaded from the project's models/ensemble_models/weighted_ensembler directory.
        """
        with open(f"{self.project_root}/models/ensemble_models/weighted_ensembler/weighted_ensembler_params.json", "r") as f:
            self.weighted_ensembler_params = json.load(f)

    
    def save_ensemble_model(self, type: str):
        if type == "weighted":
            self.weighted_ensembler.save_ensemble_model()
        else:
            raise ValueError(f"Unknown ensemble model type: {type}")
        
    def load_ensemble_model(self, type: str):
        if type == "weighted":
            return self.weighted_ensembler.load_ensemble_model()
        else:
            raise ValueError(f"Unknown ensemble model type: {type}")


