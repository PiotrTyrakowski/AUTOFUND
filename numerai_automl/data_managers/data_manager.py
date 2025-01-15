import pandas as pd
from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_downloader import DataDownloader
from numerai_automl.data_managers.data_saver import DataSaver
from numerai_automl.utils.utils import get_project_root



class DataManager:
    def __init__(self, data_version: str = "v5.0", feature_set: str = "small"):
        self.project_root = get_project_root()
        self.data_loader = DataLoader(data_version, feature_set)
        self.data_downloader = DataDownloader(data_version)
        self.data_saver = DataSaver()


    # TODO: change those method to be correct this are only for testing rn.
    def load_train_data_for_base_models(self):
        return self.data_loader.load_train_data()
    

    def load_data_for_creating_predictions_for_base_models(self):
        return self.data_loader.load_validation_data()
    
    def save_predictions_for_base_models(self, predictions: pd.DataFrame):
        self.data_saver.save_vanila_predictions_data(predictions)

    def load_vanila_predictions_data(self):
        return self.data_loader.load_vanila_predictions_data()

    def save_neutralized_predictions_for_base_models(self, neutralized_predictions: pd.DataFrame):
        self.data_saver.save_neutralized_predictions_data(neutralized_predictions)

    def load_neutralized_predictions_for_base_models(self):
        return self.data_loader.load_neutralized_predictions_data()
    
    def load_ranked_neutralized_predictions_for_base_models(self):
        all_neutralized_predictions = self.load_neutralized_predictions_for_base_models()
        cols = [col for col in all_neutralized_predictions.columns if "neutralized" in col]
        
        neutralized_predictions = all_neutralized_predictions.copy()
        neutralized_predictions[cols] = neutralized_predictions.groupby("era")[cols].rank(pct=True)

        return neutralized_predictions


    def load_validation_data_for_neutralization_of_base_models(self):
        """
        Load validation data for neutralization of base models
        this will be data frame with columns:
        - targets - all targets
        - features within the feature set
        - predictions - predictions of base models predictions_model_{target_name}
        """
        return self.data_loader.load_vanila_predictions_data()
    
    def load_train_data_for_ensembler(self):
        return self.load_ranked_neutralized_predictions_for_base_models()
    
    def load_validation_data_for_ensembler(self):
        return self.load_ranked_neutralized_predictions_for_base_models()


   
