# numerai_automl/target_ensembler.py

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import MinMaxScaler
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.config.config import LIGHTGBM_PARAM_GRID
import lightgbm as lgb

random.seed(42)
np.random.seed(42)


# OLD PLAN:
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
            :param models: dictionary with pretrained side models that will be used in ensemble by meta-model
            :param type_of_ensemble: type of ensemble method: average, weighted_average, lightgbm
            :param main_model: model that will be used as meta-model in ensemble: LGBM or
            None if type_of_ensemble is rank_mean or weighted_average
            :param list_of_weights: list of weights for weighted_average method
            """
            self.models = models
            self.type_of_ensemble = type_of_ensemble
            self.main_model = main_model
            self.list_of_weights = list_of_weights

        def predict(self, X: pd.DataFrame) -> pd.Series:
            """
            This method predicts the target variable based on the ensemble method
            :param X: df with features and era columns used to predict side targets and then based on them predict the main target
            :return: pd.Series with predictions of the main target variable
            """
            if self.type_of_ensemble == "average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X.drop(columns=["era"]))
                predictions["era"] = X["era"].reset_index(drop=True)
                return pd.Series(predictions.groupby("era").rank(pct=True).mean(axis=1))
            elif self.type_of_ensemble == "weighted_average":
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X.drop(columns=["era"]))
                predictions["era"] = X["era"].reset_index(drop=True)
                return pd.Series((predictions.groupby("era").rank(pct=True) * self.list_of_weights).sum(axis=1))
            elif self.type_of_ensemble == ("lightgbm" or "linear_regression"):
                if self.main_model is None:
                    raise ValueError(f"Main model is not defined for {self.type_of_ensemble} ensemble.")
                if "era" in X.columns: X = X.drop(columns=["era"])
                predictions = pd.DataFrame()
                for name, model in self.models.items():
                    predictions[name] = model.predict(X)
                predictions = self.main_model.predict(predictions)
                min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                return pd.Series(min_max_scaler.fit_transform(predictions.reshape(-1, 1)).ravel())
            else:
                raise ValueError(f"Invalid type of ensemble: {self.type_of_ensemble}")

        def predict_based_on_prediction_df(self, predictions: pd.DataFrame) -> pd.Series:
            """
            This method predicts the target variable based on df with predictions of the side models, This method is used in finding the best ensemble only
            :param predictions: df with predictions of the side models and era column
            :return: pd.Series with predictions of the main target variable
            """
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
                 number_of_interations: int = 30,
                 main_target: str = 'target'):
        """
        This class represents the TargetEnsembler that creates ensembles based on the given models and predictions
        :param models: dictionary with pretrained models that predicts side targets on features columns, used in creation of ensembles
        :param predictions_train: df with predictions of the side models on the training data and era column
        :param predictions: df with predictions of the side models on the validation data and era column
        :param number_of_interations: number of iterations in the ensemble methods, the bigger number the more ensemble methods will be checked
        :param main_target: name of the main target variable
        :var methods: list of ensemble methods: average, weighted_average, lightgbm, random, None
        :var scorer: instance of the Scorer class
        :var scores_of_ensembles: pd.DataFrame with scores of Ensembles: rows are names of the ensembles and columns are mean of score, std of score, sharpe ratio and max drawdown
        """
        # predictions should have era column
        assert 'era' in predictions.columns, "predictions should have era column"
        assert 'era' in predictions_train.columns, "predictions_train should have era column"
        self.models = models
        self.scorer = Scorer()
        self.main_target = main_target
        self.methods = ["average", "weighted_average", "lightgbm", "random", None]
        self.predictions_train = predictions_train
        self.predictions = predictions
        self.number_of_interations = number_of_interations
        self.scores_of_ensembles = pd.DataFrame(columns=["mean", "std", "sharpe", "max_drawdown"])

    def ensemble(self, y_train: pd.DataFrame,
                 y_val: pd.DataFrame, method: str = None) -> Dict[str, Ensemble]:
        """
        This method creates ensemble based on the given method, if method is None it creates the best ensembles in each category of ensemble methods
        :param y_train: df with target variables of the training data, used to train the meta-model in lightgbm ensemble
        :param y_val: df with target variables of the validation data, used to compute the score of the ensemble and compare them
        :param method: type of ensemble method: average, weighted_average, lightgbm, random, None, if None it will create the best ensemble in each category
        :return: dictionary with the best ensemble for each method, if method is not None and Random it will return only the best ensemble for this method
        """
        if method is None:
            return self._ensemble_all_methods(y_train, y_val, num=self.number_of_interations)
        elif method == "average":
            return {"average": self._find_ensemble_average_type(y_val, num_models=self.number_of_interations)}
        elif method == "weighted_average":
            return {
                "weighted_average": self._find_ensemble_weighted_average(y_val, num_models=self.number_of_interations)}
        elif method == "lightgbm":
            return {
                "lightgbm": self._find_ensemble_lightgbm_type(y_train, y_val, num_models=self.number_of_interations)}
        elif method == "random":
            return self._random_method(y_train, y_val)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _find_ensemble_average_type(self, y: pd.DataFrame | pd.Series,
                                    num_models: int = 30) -> Ensemble:  # TODO set default num_models to bigger number in future
        """
        This method creates ensemble based on the average method
        :param y: main target variables of the validation data, used to compute the score of the ensemble and compare them
        :param num_models: number of models to ensemble, the bigger number the more ensemble methods will be checked
        :return: ensemble based on the average method
        """
        numbers = [random.randint(1, 7) for i in range(num_models)]
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            ensemble = self.Ensemble(models=models, type_of_ensemble="average")
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.predict_based_on_prediction_df(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y)
        i = self._which_ensemble(predictions)
        self.scores_of_ensembles.index = list(self.scores_of_ensembles.index)[:-1] + ["average"]
        return list_of_ensembles[i]

    def _prepair_predictions_for_scoring(self, predictions: pd.DataFrame, y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        This method adds the main target variable and era columns to the predictions df and returns it
        :param predictions: df with predictions of the side models
        :param y: main target variables of the validation data, used to compute the score of the ensemble and compare them
        :return: df with predictions of the side models and main target variable and era column
        """
        predictions["era"] = self.predictions["era"]
        if type(y) == pd.DataFrame:
            predictions[self.main_target] = y[self.main_target].reset_index(drop=True)
        elif type(y) == pd.Series:
            predictions[self.main_target] = y.reset_index(drop=True)
        else:
            raise ValueError("y should be pd.DataFrame or pd.Series")
        return predictions

    def _which_ensemble(self, pred: pd.DataFrame) -> int:
        """
        This method chooses the best ensemble based on the score of the ensemble
        :param pred: df with scores of the ensembles
        :return: index of the best ensemble based on mean of the scores
        """
        df = self.scorer.compute_scores(pred, self.main_target)
        df2 = df.copy()
        df = df["mean"]
        i = df.sort_values(ascending=False).index[0]
        self.scores_of_ensembles = pd.concat([self.scores_of_ensembles, df2.loc[[i]]])
        if type(i) == str:
            return int(i.split("_")[1])
        else:
            i = int(i)
        return i

    def _find_ensemble_weighted_average(self, y: pd.DataFrame | pd.Series, num_models: int = 30) -> Ensemble:
        """
        This method creates ensemble based on the weighted_average method
        :param y: main target variables of the validation data, used to compute the score of the ensemble and compare them
        :param num_models: number of models to ensemble, the bigger number the more ensemble methods will be checked
        :return: ensemble based on the weighted_average method, best from num_models random ensembles
        """
        numbers = [random.randint(1, 7) for i in range(num_models)]
        list_of_ensembles = [None for _ in range(num_models)]
        predictions = pd.DataFrame()
        for i in range(num_models):
            models = self._choose_models_to_ensemble(numbers[i])
            weights = self._random_weights(len(models))
            ensemble = self.Ensemble(models=models, type_of_ensemble="weighted_average", list_of_weights=weights)
            list_of_ensembles[i] = ensemble
            predictions[f"prediction_{i}"] = ensemble.predict_based_on_prediction_df(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y)
        i = self._which_ensemble(predictions)
        self.scores_of_ensembles.index = list(self.scores_of_ensembles.index)[:-1] + ["weighted_average"]
        return list_of_ensembles[i]

    def _find_ensemble_lightgbm_type(self, y_train: pd.DataFrame, y_val: pd.DataFrame,
                                     num_models: int = 10) -> Ensemble:
        """
        This method creates ensemble based on the lightgbm method
        :param y_train: the main target variables of the training data, used to train the meta-model in lightgbm ensemble
        :param y_val: the main target variables of the validation data, used to compute the score of the ensemble and compare them
        :param num_models: number of models to ensemble, the bigger number the more ensemble methods will be checked
        :return: ensemble based on the lightgbm method, best from num_models random ensembles
        """
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
            predictions[f"prediction_{i}"] = ensemble.predict_based_on_prediction_df(self.predictions)
        predictions = self._prepair_predictions_for_scoring(predictions, y_val)
        i = self._which_ensemble(predictions)
        self.scores_of_ensembles.index = list(self.scores_of_ensembles.index)[:-1] + ["lightgbm"]
        return list_of_ensembles[i]

    def _choose_models_to_ensemble(self, number: int = 4) -> Dict[str, Any]:
        """
        This method randomly chooses side models to ensemble, used in average and weighted_average methods
        :param number: number of models to ensemble
        :return: dictionary with randomly chosen models
        """
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

    def _get_score_of_ensemble(self, ensemble: Ensemble, X: pd.DataFrame,
                               y: pd.DataFrame) -> float:  # TODO delete this, not used
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

    def _ensemble_all_methods(self, y_train: pd.DataFrame, y_val: pd.DataFrame, num: int = 30) -> Dict[str, Ensemble]:
        """
        This method creates all possible ensembles and chooses the best one from each category
        :return: dictionary with the best ensemble for each method
        """
        best_ensembles = {}
        for method in ["average", "weighted_average", "lightgbm"]:
            best_ensembles = best_ensembles | self.ensemble(y_train, y_val, method)
        self._save_scores_of_ensembles_to_json()
        return best_ensembles

    def _random_method(self, y_train: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Ensemble]:
        """
        This method chooses random ensemble method from the list of methods
        :return: dictionary with the best ensemble for the random method
        """
        which_method = random.choice(self.methods)
        return self.ensemble(y_train, y_val, which_method)

    def get_scores_of_ensembles(self) -> pd.DataFrame:
        """
        This method returns the scores of the ensembles: mean, std, sharpe ratio and max drawdown
        :return: pd.DataFrame with scores of the ensembles, rows indexes are names of the ensembles and columns are mean, std, sharpe ratio and max drawdown
        """
        if self.scores_of_ensembles.shape[0] == 0:
            raise ValueError("There are no scores of ensembles.")
        return self.scores_of_ensembles

    def _save_scores_of_ensembles_to_json(self):
        """
        This method saves the scores of the ensembles to the json file
        """
        folder_path = "scores_of_ensembles"
        path = f"{folder_path}/scores_of_ensembles.json"
        os.makedirs(folder_path, exist_ok=True)
        self.scores_of_ensembles.to_json(path)