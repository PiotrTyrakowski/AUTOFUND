from numerai_automl.visual.abstract_plot import AbstractPlot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LowHighStdPlot(AbstractPlot):
    """
    This class creates a line plot for the features with the lowest and highest standard deviation.
    data_for_visualization: The data to be visualized. This should be a pandas DataFrame in the following format:

    """

    def __init__(self, data_for_visualization: pd.DataFrame, feature_metrics: pd.DataFrame):
        super().__init__(data_for_visualization)
        self.feature_metrics = feature_metrics

    def get_plot(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(15, 5))
        per_era_corr = self.data_for_visualization
        per_era_corr[[self.feature_metrics['std'].idxmin(), self.feature_metrics['std'].idxmax()]].plot(
            ax=ax, title="Per-era Correlation of Features to the Target", xlabel="Era"
        )
        plt.legend(["lowest std", "highest std"])
        return fig
