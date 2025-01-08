import lightgbm as lgb
from typing import Dict, List, Union
import pandas as pd
from numerai_tools.scoring import neutralize


class FeatureNeutralizer:
    def __init__(self, all_features: List[str], iterations: int=10, max_number_of_features_to_neutralize: int=6, proportions: List[float]=[0.25, 0.5, 0.75, 1.0]):
        """
        Initializes the FeatureNeutralizer with the given parameters.

        :param all_features: List of all feature names in the dataset.
        :param iterations: Number of iterations for feature selection (default is 10).
        :param max_number_of_features_to_neutralize: Maximum number of features to neutralize (default is 6).
        :param proportions: List of proportions to use for neutralization (default is [0.25, 0.5, 0.75, 1.0]).
        """
        self.all_features = all_features
        self.max_number_of_features_to_neutralize = max_number_of_features_to_neutralize
        self.proportions = proportions

    def find_neutralization_features_and_proportions(self, data: pd.DataFrame, predictions: Union[pd.DataFrame, pd.Series]) -> Dict[str, Union[List[str], float]]:
        """
        Identifies the features and proportions to use for neutralization.

        :param data: DataFrame containing all features, eras, and a target column.
                     The target column should have the name of the corresponding target.
        :param predictions: DataFrame or Series containing predictions with a column named 'prediction'.

        :return: A dictionary with keys:
                 - "features_to_neutralize": List of feature names to neutralize.
                 - "proportion": The proportion to use for neutralization.
                 
                 example: 
                 {
                    "features_to_neutralize": ["feature1", "feature2"],
                    "proportion": 0.5
                 }
        """
        # Implementation needed here
        pass

    def apply_neutralization(self, predictions: pd.DataFrame, features: pd.DataFrame, proportion: float) -> pd.DataFrame:
        """
        Applies neutralization to the predictions based on the specified features and proportion.

        :param predictions: DataFrame containing the predictions to be neutralized.
        :param features: DataFrame containing the features to be used for neutralization.
        :param proportion: The proportion of the feature effect to neutralize.

        :return: A DataFrame with neutralized predictions.
        """
        # Copy predictions to avoid modifying the original data
        neutralized = predictions.copy()
        # Apply neutralization using the specified features and proportion
        neutralized = neutralize(neutralized, features[self.features_to_neutralize], proportion=proportion)
        return neutralized

  
