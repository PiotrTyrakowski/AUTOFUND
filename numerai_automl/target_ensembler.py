# numerai_automl/target_ensembler.py

from typing import Any, Dict, List
import pandas as pd

class TargetEnsembler:
    def __init__(self, models: Dict[str, Any], feature_cols: List[str]):
        self.models = models
        self.feature_cols = feature_cols

    def ensemble(self, X: pd.DataFrame, method: str = "average") -> pd.DataFrame:
        predictions = pd.DataFrame(index=X.index)
        for target, model in self.models.items():
            predictions[target] = model.predict(X[self.feature_cols])
        if method == "average":
            ensemble_pred = predictions.rank(pct=True).mean(axis=1)
        elif method == "weighted_average":
            # Define weights based on some criteria, e.g., correlation
            weights = self._compute_weights(predictions)
            ensemble_pred = (predictions * weights).sum(axis=1) / weights.sum()
        else:
            raise ValueError("Unsupported ensembling method.")
        return ensemble_pred.to_frame("prediction")

    def _compute_weights(self, predictions: pd.DataFrame) -> pd.Series:
        # Calculate correlation matrix between predictions
        corr_matrix = predictions.corr().abs()
        
        # Calculate diversity score (inverse of average correlation)
        # Lower correlation with other models = higher weight
        diversity_scores = 1 - (corr_matrix.sum() - 1) / (len(corr_matrix) - 1)
        
        # Normalize weights to sum to 1
        weights = diversity_scores / diversity_scores.sum()
        
        return weights
