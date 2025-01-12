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

        def predict(self, X: pd.DataFrame) -> pd.DataFrame:  # TODO: probably broken
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
                    predictions[name] = model.predict(X.drop(columns=["era"]))
                predictions["era"] = X["era"]
                return pd.DataFrame(predictions.groupby("era").rank(pct=True).mean(axis=1))
            elif self.type_of_ensemble == "weighted_average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions["era"] = X["era"]
                return (predictions.groupby("era").rank(pct=True) * self.list_of_weights).sum(axis=1)
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

        def easy_predict(self, predictions: pd.DataFrame) -> pd.Series:
            if self.type_of_ensemble == "average":
                columns_to_take = list(self.models.keys()) + ["era"]
                predictions = predictions[columns_to_take]
                return pd.Series(predictions.groupby("era").rank(pct=True).mean(axis=1))
            elif self.type_of_ensemble == "weighted_average":
                columns_to_take = list(self.models.keys()) + ["era"]
                predictions = predictions[columns_to_take]
                return pd.Series(
                    (predictions.groupby("era").rank(pct=True) * self.list_of_weights).sum(axis=1))
            elif self.type_of_ensemble == ("lightgbm" or "linear_regression"):
                min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                if "era" not in predictions.columns:
                    return pd.Series(
                        min_max_scaler.fit_transform(self.main_model.predict(predictions)))
                return pd.Series(
                    min_max_scaler.fit_transform(
                        self.main_model.predict(predictions.drop(labels=['era'], axis=1)).reshape(-1, 1)
                    ).ravel()
                )

        def __str__(self):
            if self.type_of_ensemble == "weighted_average":
                return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()} and weights: {self.list_of_weights}"
            if self.type_of_ensemble == "lightgbm" or self.type_of_ensemble == "linear_regression":
                return (f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()} and main model: "
                        f"{self.main_model} and parameters: {self.main_model.get_params()}")
            return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()}"

        def __repr__(self):
            if self.type_of_ensemble == "weighted_average":
                return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()} and weights: {self.list_of_weights}"
            if self.type_of_ensemble == "lightgbm" or self.type_of_ensemble == "linear_regression":
                return (f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()} and main model: "
                        f"{self.main_model} and parameters: {self.main_model.get_params()}")
            return f"Ensemble: {self.type_of_ensemble} with models: {self.models.keys()}"

        def get_models_names(self) -> List[str]:
            return list(self.models.keys())

    def __init__(self, models: Dict[str, Any], predictions_train: pd.DataFrame, predictions: pd.DataFrame,
                 main_target: str = 'target'):
        """
        This class represents the TargetEnsembler which creates ensembles based on the given models
        :param models: dictionary with models that will be used in ensembles
        :param main_target: name of the main target variable
        :param predictions_train: dataframe with predictions of the models from the models dictionary on training data
        :param predictions: dataframe with predictions of the models from the models dictionary on validation data
        :var scorer: instance of the Scorer class
        :var methods: list of ensemble methods: average, weighted_average, lightgbm, linear_regression, random, None;
        None means that it will choose all methods and user will be able to compare the results
        """
        # predictions should have era column
        assert 'era' in predictions.columns, "predictions should have era column"
        assert 'era' in predictions_train.columns, "predictions_train should have era column"
        self.models = models
        self.scorer = Scorer()
        self.main_target = main_target
        self.methods = ["average", "weighted_average", "lightgbm", "linear_regression", "random", None]
        self.predictions_train = predictions_train
        self.predictions = predictions

    def ensemble(self, y_train: pd.DataFrame,
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
            return self._ensemble_all_methods(y_train,y_val)
        elif method == "average":
            return {"average": self._average(y_val)}
        elif method == "weighted_average":
            return {"weighted_average": self._weighted_average(y_val)}
        elif method == "lightgbm":
            return {"lightgbm": self._lightgbm(y_train,  y_val)}
        elif method == "linear_regression":
            return {"linear_regression": self._linear_regression( y_train,  y_val)}
        elif method == "random":
            return self._random_method(y_train,  y_val)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _average(self, y: pd.DataFrame | pd.Series,
                 num_models: int = 30) -> Ensemble:  # TODO set default num_models to bigger number in future
        numbers = [random.randint(1, 7) for i in range(num_models)]
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            ensemble = self.Ensemble(models=models, type_of_ensemble="average")
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.easy_predict(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y)
        return list_of_ensembles[self._which_ensemble(predictions)]

    def _prepair_predictions_for_scoring(self, predictions: pd.DataFrame, y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        predictions["era"] = self.predictions["era"]
        if type(y) == pd.DataFrame:
            predictions[self.main_target] = y[self.main_target].reset_index(drop=True)
        elif type(y) == pd.Series:
            predictions[self.main_target] = y.reset_index(drop=True)
        else:
            raise ValueError("y should be pd.DataFrame or pd.Series")
        return predictions

    def _which_ensemble(self, pred: pd.DataFrame) -> int:
        df = self.scorer.compute_scores(pred, self.main_target)
        print(df)
        df = df["mean"]
        i = df.sort_values(ascending=False).index[0]
        if type(i) == str:
            return int(i.split("_")[1])
        else:
            i = int(i)
        return i

    def _weighted_average(self, y: pd.DataFrame | pd.Series, num_models: int = 30) -> Ensemble:
        numbers = [random.randint(1, 7) for i in range(num_models)]
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            weights = self._random_weights(len(models))
            print(weights, models.keys())
            ensemble = self.Ensemble(models=models, type_of_ensemble="weighted_average", list_of_weights=weights)
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.easy_predict(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y)
        return list_of_ensembles[self._which_ensemble(predictions)]

    def _lightgbm(self,  y_train: pd.DataFrame, y_val: pd.DataFrame,
                  num_models: int = 10) -> Ensemble:
        param_space = LIGHTGBM_PARAM_GRID
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            params = {k: random.choice(v) for k, v in param_space.items()}
            meta_model = lgb.LGBMRegressor(**params)
            meta_model.fit(self.predictions_train.drop(labels='era', axis=1).reset_index(drop=True),
                           y_train[self.main_target].reset_index(drop=True))
            ensemble = self.Ensemble(models=self.models, type_of_ensemble="lightgbm", main_model=meta_model)
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.easy_predict(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y_val)
        return list_of_ensembles[self._which_ensemble(predictions)]

    def _linear_regression(self,  y_train: pd.DataFrame, y_val: pd.DataFrame,
                           num_models: int = 10) -> Ensemble:
        param_space = ELASTIC_NET_PARAM_GRID
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            params = {k: random.choice(v) for k, v in param_space.items()}
            meta_model = ElasticNet(**params)
            meta_model.fit(self.predictions_train.drop(labels='era', axis=1).reset_index(drop=True),
                           y_train[self.main_target].reset_index(drop=True))
            ensemble = self.Ensemble(models=self.models, type_of_ensemble="linear_regression", main_model=meta_model)
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.easy_predict(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y_val)
        return list_of_ensembles[self._which_ensemble(predictions)]

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
