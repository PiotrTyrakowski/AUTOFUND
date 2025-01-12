# numerai_automl/target_ensembler.py

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import LIGHTGBM_PARAM_GRID, ELASTIC_NET_PARAM_GRID
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

random.seed(42)
np.random.seed(42)


# PLAN TODO:
# 1. Simple Ensemble Based on Numerai's Example Notebook
# 1a. Similar simple methods
# 2. Weighted Ensemble Based on Correlation # currently difficult to implement
# 3. LightGBM Model as Meta-Model
# 4. Linear Regression Model as Meta-Model
# 5. compare the results of the ensemble methods
# 6. Choose the best ensemble method
# 7. Give possibility to choose the ensemble method
# 8. TabPFN

class TargetEnsembler:
    class Ensemble:  # this class could inherit from Model class from sklearn
        def __init__(self, models: Dict[str, Any], type_of_ensemble: str = "rank_mean", main_model: Any = None,
                     list_of_weights: List[float] = None):
            """
            This class represents the ensemble of models with different methods
            :param models: dictionary with side models that will be used in ensemble by meta-model
            :param type_of_ensemble: type of ensemble method: average, weighted_average, lightgbm, linear_regression
            :param main_model: model that will be used as meta-model in ensemble: LGBM or ElasticNet or
            None if type_of_ensemble is rank_mean or weighted_average
            :param list_of_weights: list of weights for weighted_average method
            """
            self.models = models
            self.type_of_ensemble = type_of_ensemble
            self.main_model = main_model
            self.list_of_weights = list_of_weights

        def predict(self, X: pd.DataFrame) -> pd.DataFrame:
            """
            This method predicts the target variable based on the ensemble method which is defined in the constructor

            Parameters:
            - X (pd.DataFrame): A DataFrame containing the features and era column, used to predict the target variable.

            Returns:
            - pd.DataFrame: A DataFrame containing only the (main) target variable predictions.
            """
            if self.type_of_ensemble == "average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions["era"] = X["era"]
                return predictions.groupby("era").rank(pct=True).mean(axis=1)
            elif self.type_of_ensemble == "weighted_average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions["era"] = X["era"]
                return predictions.groupby("era").rank(pct=True).apply(lambda x: (x * self.list_of_weights).sum(axis=1))
            elif self.type_of_ensemble == ("lightgbm" or "linear_regression"):
                if self.main_model is None:
                    raise ValueError(f"Main model is not defined for {self.type_of_ensemble} ensemble.")
                X_for_main_model = X.copy()
                for name, model in self.models.items():
                    X_for_main_model[name] = model.predict(X)
                predictions = self.main_model.predict(X_for_main_model)
                min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                return min_max_scaler.fit_transform(predictions)
            else:
                raise ValueError(f"Invalid type of ensemble: {self.type_of_ensemble}")

        def __str__(self):
            return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()}"

        def __repr__(self):
            return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()}"

        def get_models_names(self) -> List[str]:
            return list(self.models.keys())

    def __init__(self, models: Dict[str, Any], main_target: str = 'target'):
        """
        This class represents the TargetEnsembler which creates ensembles based on the given models
        :param models: dictionary with models that will be used in ensembles
        :param main_target: name of the main target variable
        :var scorer: instance of the Scorer class
        :var methods: list of ensemble methods: average, weighted_average, lightgbm, linear_regression, random, None;
        None means that it will choose all methods and user will be able to compare the results
        """
        self.models = models
        self.scorer = Scorer()
        self.main_target = main_target
        self.methods = ["average", "weighted_average", "lightgbm", "linear_regression", "random", None]

    def ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                 y_val: pd.DataFrame, method: str = None) -> Dict[str, Ensemble]:
        """
        This method creates various ensembles based on the method parameter and chooses the best one from each category
        :param X_train: this dataframe contains the features and era column and predicted targets,
        used to train the models as Meta_Models
        :param y_train: this dataframe contains all target variables, used to train the models
        :param X_val: this dataframe contains the features and era column, used to validate the models
        :param y_val: this dataframe contains all target variables, used to validate the models
        :param method: this parameter defines the ensemble method
        :return: dictionary with the best ensemble for each method if method is None, otherwise it returns dict with
        the best ensemble for the given method
        """
        if method is None:
            return self._ensemble_all_methods(X_train, y_train, X_val, y_val)
        elif method == "average":
            return {"average": self._average(X_val, y_val)}
        elif method == "weighted_average":
            return {"weighted_average": self._weighted_average(X_val, y_val)}
        elif method == "lightgbm":
            return {"lightgbm": self._lightgbm(X_train, y_train, X_val, y_val)}
        elif method == "linear_regression":
            return {"linear_regression": self._linear_regression(X_train, y_train, X_val, y_val)}
        elif method == "random":
            return self._random_method(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _average(self, X: pd.DataFrame, y: pd.DataFrame, num_models: int = 30) -> Ensemble:
        best_ensemble = None
        best_score = -float('inf')
        numbers = [random.randint(1, 7) for i in range(num_models)]
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            ensemble = self.Ensemble(models=models, type_of_ensemble="average")
            score = self._get_score_of_ensemble(ensemble, X, y)
            if score > best_score:
                best_score = score
                best_ensemble = ensemble
        return best_ensemble

    def _weighted_average(self, X: pd.DataFrame, y: pd.DataFrame, num_models: int = 30) -> Ensemble:
        best_ensemble = None
        best_score = -float('inf')
        numbers = [random.randint(1, 7) for i in range(num_models)]
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            weights = self._random_weights(len(models))
            ensemble = self.Ensemble(models=models, type_of_ensemble="weighted_average", list_of_weights=weights)
            score = self._get_score_of_ensemble(ensemble, X, y)
            if score > best_score:
                best_score = score
                best_ensemble = ensemble
        return best_ensemble

    def _lightgbm(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame,
                  num_models: int = 30) -> Ensemble:
        param_space = LIGHTGBM_PARAM_GRID
        best_ensemble = None
        best_score = -float('inf')
        for i in range(num_models):
            params = {k: random.choice(v) for k, v in param_space.items()}
            meta_model = lgb.LGBMRegressor(**params)
            meta_model.fit(X_train, y_train[self.main_target])
            ensemble = self.Ensemble(models=self.models, type_of_ensemble="lightgbm", main_model=meta_model)
            score = self._get_score_of_ensemble(ensemble, X_val, y_val)
            if score > best_score:
                best_score = score
                best_ensemble = ensemble
        return best_ensemble

    def _linear_regression(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame,
                           num_models: int = 30) -> Ensemble:
        param_space = ELASTIC_NET_PARAM_GRID
        best_ensemble = None
        best_score = -float('inf')
        for i in range(num_models):
            params = {k: random.choice(v) for k, v in param_space.items()}
            meta_model = ElasticNet(**params)
            meta_model.fit(X_train, y_train[self.main_target])
            ensemble = self.Ensemble(models=self.models, type_of_ensemble="linear_regression", main_model=meta_model)
            score = self._get_score_of_ensemble(ensemble, X_val, y_val)
            if score > best_score:
                best_score = score
                best_ensemble = ensemble
        return best_ensemble

    def _choose_models_to_ensemble(self, number: int = 4) -> Dict[str, Any]:
        if number <= 0 or number > 7:
            raise ValueError("Number of models to ensemble should be between 1 and 7.")
        # Randomly choose models to ensemble
        names = random.sample(list(self.models.keys()), number)
        return {name: self.models[name] for name in names}

    @staticmethod
    def _random_weights(number: int) -> List[float]:
        """
        This method generates random weights for the weighted_average ensemble method that sum up to 1
        :param number: number of weights
        :return: list of random weights that sum up to 1
        """
        weights = [random.random() for _ in range(number)]
        return [w / sum(weights) for w in weights]

    def _get_score_of_ensemble(self, ensemble: Ensemble, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        This method computes the score of the ensemble based on the main target variable
        :param ensemble: ensemble of side models and meta-model that will be used to predict the main target variable
        :param X: features and era column
        :param y: target variables, could be all targets or only the main target
        :return: score of the ensemble based on the main target variable and main target variable predictions
        """
        y = y[self.main_target]
        predictions = ensemble.predict(X)
        predictions.columns = [f"prediction_{col}" for col in predictions.columns]
        # in fact this is one column with prediction of the main target
        predictions[self.main_target] = y
        predictions["era"] = X["era"]
        return self.scorer.compute_scores(predictions, self.main_target)["mean"].mean()

    def _ensemble_all_methods(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                              y_val: pd.DataFrame) -> Dict[str, Ensemble]:
        """
        This method creates all possible ensembles and chooses the best one from each category
        :return: dictionary with the best ensemble for each method
        """
        best_ensembles = {}
        for method in self.methods:
            best_ensembles = best_ensembles | self.ensemble(X_train, y_train, X_val, y_val, method)
        return best_ensembles

    def _random_method(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                       y_val: pd.DataFrame) -> Dict[str, Ensemble]:
        """
        This method chooses random ensemble method from the list of methods
        :return: dictionary with the best ensemble for the random method
        """
        which_method = random.choice(self.methods)
        return self.ensemble(X_train, y_train, X_val, y_val, which_method)
