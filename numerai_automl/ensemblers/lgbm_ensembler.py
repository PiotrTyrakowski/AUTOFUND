from typing import Any, Dict, List, Union
import pandas as pd
import random
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import LIGHTGBM_PARAMS_OPTION
import cloudpickle

from numerai_automl.utils.utils import get_project_root

lightgbm_param_grid = LIGHTGBM_PARAMS_OPTION

# TODO: MAKE RANDOM SEARCH FOR FINDING THE BEST PARAMETERS
class LGBMEnsembler:
    def __init__(self, 
                 all_neutralized_prediction_features: List[str],
                 target_name: str = 'target'
                 ):
        self.scorer = Scorer()
        self.all_neutralized_prediction_features = all_neutralized_prediction_features
        self.target_name = target_name
        self.neutralized_predictions_model_target = f"neutralized_predictions_model_{target_name}"
        self.model_trainer = None # LGBMModelTrainer(lightgbm_param_grid)
        self.project_root = get_project_root()
    
    def find_lgbm_ensemble(self, train_data: pd.DataFrame):
        # check if neutralized_predictions_model_target is in the train_data
        assert self.neutralized_predictions_model_target in train_data.columns, f"The feature {self.neutralized_predictions_model_target} is not in the validation data"

        # check if all_neutralized_prediction_features are in the train_data
        for feature in self.all_neutralized_prediction_features:
            assert feature in train_data.columns, f"The feature {feature} is not in the validation data"

        # check if era is in the train_data
        assert "era" in train_data.columns, "The validation data does not have an era column"

        # check if target_name is in the validation_data
        assert self.target_name in train_data.columns, f"The target {self.target_name} is not in the validation data"

        X = train_data[self.all_neutralized_prediction_features]
        y = train_data[self.target_name]

        self.model_trainer = LGBMModelTrainer(lightgbm_param_grid)

        self.model_trainer.train(X, y)

        self.model_trainer.get_model()

    def predict(self, X: pd.DataFrame):
        return self.model_trainer.get_model().predict(X)
    
    def save_ensemble_model(self):
        assert self.model_trainer is not None, "The ensemble features and weights are not loaded"

        assert self.model_trainer.is_trained, "The model is not trained"
        
        # save model with pickl
        with open(f"{self.project_root}/models/ensemble_models/lgbm_ensembler/lgbm_ensembler.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load_ensemble_model(cls):
        project_root = get_project_root()
        with open(f"{project_root}/models/ensemble_models/lgbm_ensembler/lgbm_ensembler.pkl", "rb") as f:
            return cloudpickle.load(f)