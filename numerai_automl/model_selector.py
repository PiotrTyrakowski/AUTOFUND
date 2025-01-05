# numerai_automl/model_selector.py

import pandas as pd

class ModelSelector:
    def __init__(self, scores: pd.DataFrame):
        self.scores = scores

    def select_best_by_corr(self) -> str:
        return self.scores['corr'].idxmax()

    def select_best_by_mmc(self) -> str:
        return self.scores['mmc'].idxmax()

    def select_best_combined(self, weight_corr: float = 0.5, weight_mmc: float = 0.5) -> str:
        combined_score = self.scores['corr'] * weight_corr + self.scores['mmc'] * weight_mmc
        return combined_score.idxmax()
