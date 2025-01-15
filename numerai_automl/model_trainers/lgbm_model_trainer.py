# numerai_automl/model_trainer.py

import lightgbm as lgb
from typing import Dict, List
import pandas as pd
from numerai_automl.model_trainers.abstract_model_trainer import AbstractModelTrainer
class LGBMModelTrainer(AbstractModelTrainer):
    def __init__(self, params: Dict):
        self.params = params
        self.model = lgb.LGBMRegressor(**self.params)
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_trained = True
        
    def get_model(self):
        if not self.is_trained:
            raise Exception("Model is not trained")
        return self.model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise Exception("Model is not trained")
        return self.model.predict(X)
