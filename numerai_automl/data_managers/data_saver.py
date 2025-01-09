import pandas as pd
from numerai_automl.utils.utils import get_project_root


class DataSaver:
    def __init__(self):
        self.project_root = get_project_root()

    def save_vanila_predictions_data(self, predictions: pd.DataFrame):
        """Save all required dataset files"""
        filepath = f"{self.project_root}/predictions/vanila_predictions.parquet"
        predictions.to_parquet(filepath)

    def save_neutralized_predictions_data(self, predictions: pd.DataFrame):
        filepath = f"{self.project_root}/predictions/neutralized_predictions.parquet"
        predictions.to_parquet(filepath)

    def save_ensembled_predictions_data(self, predictions: pd.DataFrame):
        filepath = f"{self.project_root}/predictions/ensembled_predictions.parquet"
        predictions.to_parquet(filepath)

