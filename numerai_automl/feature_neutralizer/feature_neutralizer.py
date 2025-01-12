import random
import lightgbm as lgb
from typing import Dict, List, Union
import pandas as pd
from numerai_tools.scoring import neutralize
from numerai_automl.scorer.scorer import Scorer


class FeatureNeutralizer:
    def __init__(self, all_features: List[str], target_name: str, iterations: int=10, max_number_of_features_to_neutralize: int=6, proportions: List[float]=[0.25, 0.5, 0.75, 1.0]):
        """
        Initializes the FeatureNeutralizer with the given parameters.

        :param all_features: List of all feature names in the dataset.
        :param iterations: Number of iterations for feature selection (default is 10).
        :param max_number_of_features_to_neutralize: Maximum number of features to neutralize (default is 6).
        :param proportions: List of proportions to use for neutralization (default is [0.25, 0.5, 0.75, 1.0]).
        """
        self.iterations = iterations
        self.all_features = all_features
        self.max_number_of_features_to_neutralize = max_number_of_features_to_neutralize
        self.proportions = proportions
        self.scorer = Scorer()
        self.target_name = target_name
        
    def find_neutralization_features_and_proportions(self, data: pd.DataFrame, predictions: Union[pd.DataFrame, pd.Series], metric: str="mean") -> Dict[str, Union[Dict, Dict]]:
        """
        Identifies the features and proportions to use for neutralization.

        :param data: DataFrame containing all features, eras, and a target column.
                     The target column should have the name of the corresponding target.
        :param predictions: DataFrame or Series containing predictions with a column named 'prediction'.
        :param metric: The metric to use for scoring the neutralization. Default is "mean".
                       Can be "mean", "std", "sharpe", "max_drawdown".

        :return: A dictionary with keys:
                 - "neutralization_params": Dictionary containing:
                    - "features_to_neutralize": List of feature names to neutralize
                    - "proportion": The proportion to use for neutralization
                 - "scores": Dictionary containing the scoring metrics (mean, sharpe, std, max_drawdown)
                 
                 example: 
                 {
                    "neutralization_params": {
                        "features_to_neutralize": ["feature1", "feature2"],
                        "proportion": 0.5
                    },
                    "scores": {
                        "mean": 0.025,
                        "sharpe": 1.5,
                        "std": 0.015,
                        "max_drawdown": -0.1
                    }
                 }
        """


        # check if the data has the same ids as the predictions
        assert (data.index.equals(predictions.index)), "The data and predictions do not have the same ids"

        assert metric in ["mean", "std", "sharpe", "max_drawdown"], "The metric is not valid"

        # this will be used to compute the neutralization
        data_with_predictions = data.copy()
        data_with_predictions["prediction"] = predictions

        # this will be used to store the neutralized predictions
        neutralized_predictions = data[["era", self.target_name]].copy()

        # this will be to save features and proportions
        # key is "neutralized_predictions_{i}" and value is dict with keys "features_to_neutralize" and "proportion"
        features_and_proportions = {}

        for i in range(self.iterations):
            number_of_features_to_neutralize = random.randint(1, self.max_number_of_features_to_neutralize)
            features_to_neutralize = random.sample(self.all_features, number_of_features_to_neutralize)
            proportion = random.choice(self.proportions)

            neutralized = data_with_predictions.groupby("era", group_keys=True).apply(
                lambda d: neutralize(
                d[["prediction"]],
                d[features_to_neutralize],
                proportion=proportion
                )
            ).reset_index().set_index("id")

            neutralized_predictions[f"neutralized_predictions_{i}"] = neutralized["prediction"]

            features_and_proportions[f"neutralized_predictions_{i}"] = {
                "features_to_neutralize": features_to_neutralize,
                "proportion": proportion
            }
        
        scores = self.scorer.compute_scores(neutralized_predictions, self.target_name)

        # sort the scores by the metric

        if metric == "mean" or metric == "sharpe":
            scores = scores.sort_values(by=metric, ascending=False)
        elif metric == "std" or metric == "max_drawdown":
            scores = scores.sort_values(by=metric, ascending=True)

        # Get the best performing neutralization configuration
        best_prediction_column = scores.index[0]
        best_neutralization_params = features_and_proportions[best_prediction_column]
        best_scores = scores.loc[best_prediction_column].to_dict()

        return {
            "neutralization_params": best_neutralization_params,
            "scores": best_scores
        }

    # TODO: THIS WILL BE FOR THE LIVE DATA I NEED TO REPAIR IT TO WORK. BECAUSE LIVE DATA DOES NOT HAVE ERA.
    def apply_neutralization(self, predictions: pd.DataFrame, features_with_era: pd.DataFrame, proportion: float) -> pd.DataFrame:
        """
        Applies neutralization to the predictions based on the specified features and proportion.

        :param predictions: DataFrame containing the predictions to be neutralized.
        :param features: DataFrame containing the features to be used for neutralization with era column.
        :param proportion: The proportion of the feature effect to neutralize.

        :return: A DataFrame with neutralized predictions.
        """

        # check if the predictions have column named "prediction"
        assert "prediction" in predictions.columns, "The predictions do not have a column named 'prediction'"

        # check if the predictions and features have the same ids
        assert (predictions["id"] == features_with_era["id"]).all(), "The predictions and features do not have the same ids"

        # create new dataframe with predictions and features
        data = pd.concat([predictions, features_with_era], axis=1)


        # Apply neutralization using the specified features and proportion
        neutralized = neutralize(neutralized, features, proportion=proportion)
        return neutralized

  
