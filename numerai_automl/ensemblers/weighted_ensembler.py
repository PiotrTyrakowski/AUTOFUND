from typing import Any, Dict, List, Union
import pandas as pd
import random
from numerai_automl.scorer.scorer import Scorer

class WeightedTargetEnsembler:
    def __init__(self, 
                 all_neutralized_prediction_features: List[str],
                 target_name: str = 'target',
                 number_of_interations: int = 30,
                 max_number_of_prediction_features_for_ensemble: int = 5,
                 number_of_diffrent_weights_for_ensemble: int = 5
                 ):
        """
        Initialize the WeightedTargetEnsembler for combining multiple prediction models.
        
        Parameters:
            all_neutralized_prediction_features: List of feature names containing neutralized predictions
            target_name: Name of the target variable column
            number_of_interations: Number of ensemble combinations to try
            max_number_of_prediction_features_for_ensemble: Maximum number of features to include in ensemble
            number_of_diffrent_weights_for_ensemble: Number of different weight combinations to try per iteration
        """
        self.scorer = Scorer()
        self.all_neutralized_prediction_features = all_neutralized_prediction_features
        self.target_name = target_name
        self.neutralized_predictions_model_target = "neutralized_predictions_model_target"
        self.number_of_interations = number_of_interations
        self.max_number_of_prediction_features_for_ensemble = max_number_of_prediction_features_for_ensemble
        self.number_of_diffrent_weights_for_ensemble = number_of_diffrent_weights_for_ensemble

        assert self.max_number_of_prediction_features_for_ensemble <= len(self.all_neutralized_prediction_features), "The max number of prediction features for ensemble is greater than the number of all neutralized prediction features"

    
    def find_ensemble_prediction_features_and_proportions(self, validation_data: pd.DataFrame, metric: str="mean") -> Dict[str, Union[Dict, Dict]]:
        """
        Find optimal combination of prediction features and their weights for ensemble.
        
        Parameters:
            validation_data: DataFrame containing:
                - Neutralized prediction features
                - 'era' column for time-based validation
                - Target variable column
            metric: Optimization metric ('mean', 'std', 'sharpe', or 'max_drawdown')
        Returns:
            Dictionary containing:
                - neutralization_params: Best feature combination and weights
                - scores: Performance metrics for the best ensemble
                
                example:
                {
                    "neutralization_params": {
                        "neutralized_prediction_features": ["neutralized_predictions_model_target", "neutralized_predictions_model_target_victor_20"],
                        "weights": [0.6, 0.4]
                    },
                    "scores": {
                        "mean": 0.025,
                        "sharpe": 1.5, 
                        "std": 0.015,
                        "max_drawdown": -0.1
                    }
                }
        """

        # check if metric is valid  
        assert metric in ["mean", "std", "sharpe", "max_drawdown"], "The metric is not valid"

        # check if neutralized_predictions_model_target is in the validation_data
        assert self.neutralized_predictions_model_target in validation_data.columns, f"The feature {self.neutralized_predictions_model_target} is not in the validation data"

        # check if all_neutralized_prediction_features are in the validation_data
        for feature in self.all_neutralized_prediction_features:
            assert feature in validation_data.columns, f"The feature {feature} is not in the validation data"

        # check if era is in the validation_data
        assert "era" in validation_data.columns, "The validation data does not have an era column"

        # check if target_name is in the validation_data
        assert self.target_name in validation_data.columns, f"The target {self.target_name} is not in the validation data"

        validation_data = validation_data[self.all_neutralized_prediction_features + ["era", self.target_name]]

        neutralized_prediction_features_and_weights = {}

        neutralized_prediction_features_and_weights[f"ensemble_predictions_{0}"] = {
            "neutralized_prediction_features": [self.neutralized_predictions_model_target],
            "weights": [1]
        }

        ensemble_predictions_df = validation_data[["era", self.target_name]].copy()
        ensemble_predictions_df[f"ensemble_predictions_{0}"] = validation_data[self.neutralized_predictions_model_target]

        prediction_features_without_main_prediction = self.all_neutralized_prediction_features.copy()
        prediction_features_without_main_prediction.remove(self.neutralized_predictions_model_target)

        for i in range(self.number_of_interations):
            number_of_prediction_features_for_ensemble = random.randint(1, self.max_number_of_prediction_features_for_ensemble - 1)
            prediction_features_for_ensemble = random.sample(prediction_features_without_main_prediction, number_of_prediction_features_for_ensemble)
            prediction_features_for_ensemble.append(self.neutralized_predictions_model_target)

            weights_for_ensemble = self._mean_weights(len(prediction_features_for_ensemble))
            weights_series = pd.Series(weights_for_ensemble, index=prediction_features_for_ensemble)
            ensemble_predictions_df[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble}"] = (validation_data[prediction_features_for_ensemble] * weights_series).sum(axis=1)
            
            neutralized_prediction_features_and_weights[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble}"] = {
                "neutralized_prediction_features": prediction_features_for_ensemble,
                "weights": weights_series
            }
            
            for j in range(1, self.number_of_diffrent_weights_for_ensemble):
                weights_for_ensemble = self._random_weights(len(prediction_features_for_ensemble))
                weights_series = pd.Series(weights_for_ensemble, index=prediction_features_for_ensemble)
                ensemble_predictions_df[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble + j}"] = (validation_data[prediction_features_for_ensemble] * weights_series).sum(axis=1)
                
                neutralized_prediction_features_and_weights[f"ensemble_predictions_{1 + i * self.number_of_diffrent_weights_for_ensemble + j}"] = {
                    "neutralized_prediction_features": prediction_features_for_ensemble,
                    "weights": weights_series
                }

        scores = self.scorer.compute_scores(ensemble_predictions_df, self.target_name)

        print(scores)

        if metric == "mean" or metric == "sharpe":
            scores = scores.sort_values(by=metric, ascending=False)
        elif metric == "std" or metric == "max_drawdown":
            scores = scores.sort_values(by=metric, ascending=True)

        best_prediction_column = scores.index[0]
        best_neutralization_params = neutralized_prediction_features_and_weights[best_prediction_column]
        best_scores = scores.loc[best_prediction_column].to_dict()

        return {
            "neutralization_params": best_neutralization_params,
            "scores": best_scores
        }


    
    def _mean_weights(self, number_of_weights: int) -> List[float]:
        """
        Generate equally distributed weights that sum to 1.
        
        Parameters:
            number_of_weights: Number of weights to generate
        
        Returns:
            List of equal weights (1/n for each weight)
        """
        return [1 / number_of_weights for _ in range(number_of_weights)]
    
    def _random_weights(self, number_of_weights: int) -> List[float]:
        """
        Generate random weights that sum to 1 for ensemble experimentation.
        
        Parameters:
            number_of_weights: Number of weights to generate
        
        Returns:
            List of normalized random weights that sum to 1
        """

        arr = [random.uniform(0, 1) for _ in range(number_of_weights)]
        return [ weight / sum(arr) for weight in arr]


    