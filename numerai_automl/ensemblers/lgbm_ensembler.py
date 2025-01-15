from typing import Any, Dict, List, Union
import pandas as pd
import random
from numerai_automl.model_trainers.lgbm_model_trainer import LGBMModelTrainer
from numerai_automl.scorer.scorer import Scorer

class LGBMEnsembler:
    def __init__(self, 
                 all_neutralized_prediction_features: List[str],
                 target_name: str = 'target'
                 ):
        self.scorer = Scorer()
        self.all_neutralized_prediction_features = all_neutralized_prediction_features
        self.target_name = target_name
        self.neutralized_predictions_model_target = f"neutralized_predictions_model_{target_name}"
        self.model_trainer = LGBMModelTrainer()
