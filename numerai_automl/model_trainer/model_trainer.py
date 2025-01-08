# numerai_automl/model_trainer.py

import lightgbm as lgb
from typing import Dict, List
import pandas as pd

# TODO: train using k-fold cross validation
class ModelTrainer:
    def __init__(self, params: Dict):
        self.params = params
        self.model = lgb.LGBMRegressor(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self.model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
