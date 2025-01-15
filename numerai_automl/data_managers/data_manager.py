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


    # firstly load train data
    def load_train_data_for_base_models(self):
        return self.data_loader.load_train_data()
    

    # all data for creating predicitons maybe all 
    # here we load all validation data
    def load_data_for_creating_predictions_for_base_models(self):

        return self.data_loader.load_validation_data()
    
    def save_vanila_predictions_for_base_models(self, predictions: pd.DataFrame):
        self.data_saver.save_vanila_predictions_data(predictions)

    def load_vanila_predictions_data_by_base_models(self):
        return self.data_loader.load_vanila_predictions_data()



    # all validation data for neutralization
    def save_neutralized_predictions_by_base_models(self, neutralized_predictions: pd.DataFrame):
        self.data_saver.save_neutralized_predictions_data(neutralized_predictions)

    def load_neutralized_predictions_by_base_models(self):
        return self.data_loader.load_neutralized_predictions_data()
    

    def load_ranked_neutralized_predictions_by_base_models(self):
        all_neutralized_predictions = self.load_neutralized_predictions_by_base_models()
        cols = [col for col in all_neutralized_predictions.columns if "neutralized" in col]
        
        neutralized_predictions = all_neutralized_predictions.copy()
        neutralized_predictions[cols] = neutralized_predictions.groupby("era")[cols].rank(pct=True)

        return neutralized_predictions

    def _get_min_and_max_era(self, train_data: pd.DataFrame):
        min_era = train_data["era"].min()
        max_era = train_data["era"].max()
        return min_era, max_era
    
    def load_train_data_for_ensembler(self):
        # take all ares in data and take first half of them
        train_data = self.load_train_data_for_base_models()

        min_era, max_era = self._get_min_and_max_era(train_data)

        train_data = train_data[train_data["era"] < (min_era + max_era) / 2]
        return train_data
    
    def load_validation_data_for_ensembler(self):
        validation_data = self.load_data_for_creating_predictions_for_base_models()

        min_era, max_era = self._get_min_and_max_era(validation_data)

        validation_data = validation_data[validation_data["era"] >= (min_era + max_era) / 2]
        return validation_data


   
