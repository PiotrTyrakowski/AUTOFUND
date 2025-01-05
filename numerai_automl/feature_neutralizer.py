# numerai_automl/feature_neutralizer.py

from typing import List
import pandas as pd
import numpy as np
from numerai_tools.scoring import neutralize

class FeatureNeutralizer:

    def __init__(self, neutralizer_config: dict):
        """
        Initialize with a dictionary mapping features to their neutralization proportions
        Example: {0.25: ["feature1", "feature2"], 0.5: ["feature3"], 1.0: ["feature4", "feature5"]}
        """
        self.neutralizer_config = neutralizer_config

    def apply(self, predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        neutralized = predictions.copy()
        
        # Apply neutralization for each feature group separately
        for proportion, features_list in self.neutralizer_config.items():
            if isinstance(features_list, str):
                features_list = [features_list]
            
            neutralized = neutralize(
                neutralized,
                features[features_list],
                proportion=proportion
            )
        
        return neutralized
