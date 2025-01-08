# numerai_automl/data_loader.py

import pandas as pd
from numerapi import NumerAPI
import json
from typing import List
import os

class DataLoader:
    def __init__(self, data_version: str = "v5.0", download_data: bool = False):
        self.napi = NumerAPI()
        self.data_version = data_version
        self.download_data = download_data
        
        if self.download_data:
            self.download_all_data()
            
        self.feature_metadata = self._load_feature_metadata()
        self.feature_sets = self.feature_metadata["feature_sets"]

    def download_all_data(self):
        """Download all required dataset files"""
        files = ["features.json", "train.parquet", "validation.parquet", "live.parquet"]
        for file in files:
            self._download_file(file)

    def _download_file(self, filename: str):
        """Helper method to download a single file"""
        filepath = f"{self.data_version}/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            self.napi.download_dataset(filepath)

    def download_train_data(self):
        self._download_file("train.parquet")

    def download_validation_data(self):
        self._download_file("validation.parquet")

    def download_live_data(self):
        self._download_file("live.parquet")

    def _load_feature_metadata(self) -> dict:
        if self.download_data:
            self._download_file("features.json")
        elif not os.path.exists(f"{self.data_version}/features.json"):
            raise FileNotFoundError(f"Features file not found at {self.data_version}/features.json")
            
        with open(f"{self.data_version}/features.json") as f:
            return json.load(f)

    # downsample_step is the number of eras to skip 4 is by default because 4 eras is 20 days (as per target)
    def load_train_data(self, feature_set: str = "medium", target_set: List[str] = ["target"], downsample_step: int = 4) -> pd.DataFrame:
        features = self.feature_sets[feature_set]
        filepath = f"{self.data_version}/train.parquet"
        
        if not os.path.exists(filepath):
            if self.download_data:
                self.download_train_data()
            else:
                raise FileNotFoundError(f"Train data not found at {filepath}")
                
        train = pd.read_parquet(filepath, columns=["era"] + target_set + features)
        return self._downsample_data(train, downsample_step)

    def load_validation_data(self, feature_set: str = "medium", target_set: List[str] = ["target"], downsample_step: int = 4) -> pd.DataFrame:
        features = self.feature_sets[feature_set]
        filepath = f"{self.data_version}/validation.parquet"
        
        if not os.path.exists(filepath):
            if self.download_data:
                self.download_validation_data()
            else:
                raise FileNotFoundError(f"Validation data not found at {filepath}")
                
        validation = pd.read_parquet(filepath, columns=["era", "data_type"] + target_set + features)
        validation = validation[validation["data_type"] == "validation"]
        return self._downsample_data(validation, downsample_step)

    def load_live_data(self, feature_set: str = "medium") -> pd.DataFrame:
        features = self.feature_sets[feature_set]
        filepath = f"{self.data_version}/live.parquet"
        
        if not os.path.exists(filepath):
            if self.download_data:
                self.download_live_data()
            else:
                raise FileNotFoundError(f"Live data not found at {filepath}")
                
        return pd.read_parquet(filepath, columns=features)

    def _downsample_data(self, df: pd.DataFrame, downsample_step: int) -> pd.DataFrame:
        """Helper method to downsample data by era"""
        if downsample_step > 1:
            unique_eras = df["era"].unique()[::downsample_step]
            return df[df["era"].isin(unique_eras)]
        return df
