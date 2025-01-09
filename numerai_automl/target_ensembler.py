# numerai_automl/target_ensembler.py

from typing import Any, Dict, List
import pandas as pd
import random #TODO: add seed


# PLAN TODO:
# 1. Simple Ensemble Based on Numerai's Example Notebook
# 1a. Similar simple methods
# 2. Weighted Ensemble Based on Correlation # currently difficult to implement
# 3. LightGBM Model as Meta-Model
# 4. Linear Regression Model as Meta-Model
# 5. compare the results of the ensemble methods
# 6. Choose the best ensemble method
# 7. Give possibility to choose the ensemble method
class TargetEnsembler:
    class Ensemble:
        def __init__(self, models: Dict[str, Any], type_of_ensemble: str = "rank_mean", main_model: Any = None,
                     list_of_weights: List[float] = None):
            self.models = models
            self.type_of_ensemble = type_of_ensemble
            self.main_model = main_model
            # main model is model instance if type_of_ensemble is model_LGBM or model_LR and
            # None if type_of_ensemble is rank_mean or weighted_average
            self.list_of_weights = list_of_weights # list of weights for weighted_average method
        def predict(self, X: pd.DataFrame) -> pd.DataFrame:
            if self.type_of_ensemble is "average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions["era"] = X["era"]
                return predictions.groupby("era").rank(pct=True).mean(axis=1)
            elif self.type_of_ensemble is "weighted_average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions["era"] = X["era"]
                return predictions.groupby("era").apply(lambda x: (x * self.list_of_weights).sum(axis=1))
            else:
                raise ValueError(f"Invalid type of ensemble: {self.type_of_ensemble}")


    def __init__(self, models: Dict[str, Any], main_target_era_df: pd.DataFrame, Scorer: Any):
        self.models = models
        self.scorer = Scorer
        self.main_target_era=main_target_era_df
        self.methods= ["average", "weighted_average", "lightgbm", "linear_regression", "random", None]
        # None means that it will choose all methods and user will be able to compare the results

    def ensemble(self, predictions: pd.DataFrame, method: str = None) -> Ensemble:
        # df_train and df_val have got columns with predictions from models probably
        if method is None:
            return self._ensemble_all_methods(predictions) #TODO: implement this method
        elif method == "average":
            return self._average(predictions)
        elif method == "weighted_average":
            return self._weighted_average(predictions)
        elif method == "lightgbm":
            return self._lightgbm(predictions)
        elif method == "linear_regression":
            return self._linear_regression(predictions)
        elif method == "random":
            return self._random_method(predictions)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _average(self, predictions: pd.DataFrame, num_models: int = 30) -> Ensemble:
        scores = []
        ensembles = []
        numbers = [random.randint(1, 7) for i in range(num_models)]
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            ensemble = self.Ensemble(models=models, type_of_ensemble="average")
            score = self._get_score_of_ensemble(ensemble, predictions)
            scores.append(score)
        return ensembles[scores.index(max(scores))]
    def _weighted_average(self, predictions: pd.DataFrame, num_models: int = 30) -> Ensemble:
        scores = []
        ensembles = []
        numbers = [random.randint(1, 7) for i in range(num_models)]
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            weights = self._random_weights(len(models))
            ensemble = self.Ensemble(models=models, type_of_ensemble="weighted_average", list_of_weights=weights)
            score = self._get_score_of_ensemble(ensemble, predictions)
            scores.append(score)
        return ensembles[scores.index(max(scores))]

    def _choose_models_to_ensemble(self, number: int = 4) -> Dict[str, Any]:
        if number <= 0 or number > 7:
            raise ValueError("Number of models to ensemble should be between 1 and 7.")
        # Randomly choose models to ensemble
        names = random.sample(list(self.models.keys()), number)
        return {name: self.models[name] for name in names}

    @staticmethod
    def _random_weights(number: int) -> List[float]:
        # Generate random weights that sum to 1
        weights = [random.random() for _ in range(number)]
        return [w / sum(weights) for w in weights]

    def _get_score_of_ensemble(self, ensemble: Ensemble, predictions: pd.DataFrame) -> float:
        return self.scorer.score(ensemble.predict(predictions, self.main_target_era)) # TODO: I need to change it
        # after someone will implement the scorer


