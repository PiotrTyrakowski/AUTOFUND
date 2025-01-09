# numerai_automl/data_loader.py

import pandas as pd
from numerapi import NumerAPI
import json
from typing import List
import os
from numerai_automl.utils.utils import get_project_root

# TODO: Change it to be place where we can get data how we want it

class DataLoader:
    def __init__(self, data_version: str = "v5.0", feature_set: str = "medium"):

        self.data_version = data_version
        self.project_root = get_project_root()
        self.feature_metadata = self._load_feature_metadata()
        self.feature_sets = self.feature_metadata["feature_sets"]
        self.features = self.feature_sets[feature_set]
   

    def _load_feature_metadata(self) -> dict:
       
        if not os.path.exists(f"{self.project_root}/{self.data_version}/features.json"):
            raise FileNotFoundError(f"Features file not found at {self.project_root}/{self.data_version}/features.json")

        with open(f"{self.project_root}/{self.data_version}/features.json") as f:
            return json.load(f)

    # downsample_step is the number of eras to skip 4 is by default because 4 eras is 20 days (as per target)
    def load_train_data(self, target_set: List[str] = ["target"], downsample_step: int = 4, start_era: int = 0) -> pd.DataFrame:
        filepath = f"{self.project_root}/{self.data_version}/train.parquet"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Train data not found at {filepath}")
                
        train = pd.read_parquet(filepath, columns=["era"] + target_set + self.features)
        return self._downsample_data(train, downsample_step, start_era)

    def load_validation_data(self, feature_set: str = "medium", target_set: List[str] = ["target"], downsample_step: int = 4, start_era: int = 0) -> pd.DataFrame:
        features = self.feature_sets[feature_set]
        filepath = f"{self.project_root}/{self.data_version}/validation.parquet"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Validation data not found at {filepath}")
                
        validation = pd.read_parquet(filepath, columns=["era", "data_type"] + target_set + features)
        validation = validation[validation["data_type"] == "validation"]
        return self._downsample_data(validation, downsample_step, start_era)

    def load_live_data(self, feature_set: str = "medium") -> pd.DataFrame:
        features = self.feature_sets[feature_set]
        filepath = f"{self.project_root}/{self.data_version}/live.parquet"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Live data not found at {filepath}")
                
        return pd.read_parquet(filepath, columns=features)

    def load_vanila_predictions_data(self) -> pd.DataFrame:
        filepath = f"{self.project_root}/predictions/vanila_predictions.parquet"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vanila prediction data not found at {filepath}")
        return pd.read_parquet(filepath)
    
    def load_neutralized_predictions_data(self) -> pd.DataFrame:
        filepath = f"{self.project_root}/predictions/neutralized_predictions.parquet"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Neutralized prediction data not found at {filepath}")
        return pd.read_parquet(filepath)
    
    def load_ensembled_predictions_data(self) -> pd.DataFrame:
        filepath = f"{self.project_root}/predictions/ensembled_predictions.parquet"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ensembled prediction data not found at {filepath}")
        return pd.read_parquet(filepath)

    def _downsample_data(self, df: pd.DataFrame, downsample_step: int, start_era: int) -> pd.DataFrame:
        """Helper method to downsample data by era"""
        if downsample_step > 1:
            unique_eras = df["era"].unique()[start_era::downsample_step]
            return df[df["era"].isin(unique_eras)]
        return df
