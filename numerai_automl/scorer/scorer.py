# numerai_automl/scorer.py

from typing import Dict
import pandas as pd
from numerai_tools.scoring import numerai_corr, correlation_contribution

class Scorer:
    def __init__(self):
        pass

    def compute_scores(self, data: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        Computes various scoring metrics for predictions.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing the following columns:
            - 'era': The era of the data.
            - 'predictions': The predicted values.
            - 'target': The actual target values.
        - target_name (str): The name of the target column.

        Returns:
        - pd.DataFrame: A DataFrame with the following columns:
            - 'mean': The mean of the predictions.
            - 'std': The standard deviation of the predictions.
            - 'sharpe': The Sharpe ratio of the predictions.
            - 'max_drawdown': The maximum drawdown of the predictions.

        The DataFrame will have different rows for each set of predictions, 
        with metrics calculated for each.
        """

        prediction_cols = [col for col in data.columns if "prediction" in col]

        correlations = data.groupby("era").apply(
            lambda d: numerai_corr(d[prediction_cols], d[target_name])
        )
        cumsum_corrs = correlations.cumsum()

        def get_summary_metrics(scores, cumsum_scores):
            summary_metrics = {}
            # per era correlation between predictions of the model trained on this target and cyrus
            mean = scores.mean()
            std = scores.std()
            sharpe = mean / std
            rolling_max = cumsum_scores.expanding(min_periods=1).max()
            max_drawdown = (rolling_max - cumsum_scores).max()
            return {
                "mean": mean,
                "std": std,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
            }
        target_summary_metrics = {}

        for pred_col in prediction_cols:
            target_summary_metrics[pred_col] = get_summary_metrics(
                correlations[pred_col], cumsum_corrs[pred_col]
            )
        pd.set_option('display.float_format', lambda x: '%f' % x)
        summary = pd.DataFrame(target_summary_metrics).T

        return summary
    