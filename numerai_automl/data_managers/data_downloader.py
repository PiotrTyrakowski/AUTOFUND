# numerai_automl/data_loader.py

import pandas as pd
from numerapi import NumerAPI
import json
import os
from numerai_automl.utils.utils import get_project_root


class DataDownloader:
    def __init__(self, data_version: str = "v5.0"):
        self.napi = NumerAPI()
        self.data_version = data_version
        self.project_root = get_project_root()

        
    def download_all_data(self):
        """Download all required dataset files"""
        files = ["features.json", "train.parquet", "validation.parquet", "live.parquet"]
        for file in files:
            self._download_file(file)

    def _download_file(self, filename: str):
        """Helper method to download a single file"""
        filepath = f"{self.project_root}/{self.data_version}/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            self.napi.download_dataset(filepath)

    def download_train_data(self):
        self._download_file("train.parquet")

    def download_validation_data(self):
        self._download_file("validation.parquet")

    def download_live_data(self):
        self._download_file("live.parquet")
