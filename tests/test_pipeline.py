from numerai_automl.data_managers.data_loader import DataLoader
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.scorer.scorer import Scorer


def test_pipeline():
    data_manager = DataManager(data_version="v5.0", feature_set="small")
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager.train_base_models()

    model_manager.save_base_models()

    model_manager.create_predictions_by_base_models()

    model_manager.find_neutralization_features_and_proportions_for_base_models()

    model_manager.create_neutralized_predictions_by_base_models_predictions()

    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    ensemble_manager.find_weighted_ensemble()
    ensemble_manager.find_lgbm_ensemble()


    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    predictor_weighted = meta_manager.save_predictor("weighted")
    predictor_lgbm = meta_manager.save_predictor("lgbm")

    X = data_manager.load_validation_data_for_ensembler()

    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)

    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    print(preds_weighted)
    print(preds_lgbm)



def test_pipeline2():
    data_manager = DataManager(data_version="v5.0", feature_set="medium")
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    model_manager.load_base_models()

    model_manager.create_predictions_by_base_models()

    model_manager.find_neutralization_features_and_proportions_for_base_models()

    model_manager.create_neutralized_predictions_by_base_models_predictions()

    ensemble_manager = EnsembleModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    ensemble_manager.find_weighted_ensemble()
    ensemble_manager.find_lgbm_ensemble()


    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    meta_manager.save_predictor("weighted")
    meta_manager.save_predictor("lgbm")

    predictor_weighted = meta_manager.load_predictor("weighted")
    predictor_lgbm = meta_manager.load_predictor("lgbm")

    # this will be end validation data that we will do scoring plots etc staff like that.
    X = data_manager.load_validation_data_for_ensembler()



    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)

    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    print(preds_weighted)
    print(preds_lgbm)


def test_pipeline3():
    data_loader = DataLoader(data_version="v5.0", feature_set="medium")
    data_manager = DataManager(data_version="v5.0", feature_set="medium")
    meta_manager = MetaModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )
    model_manager = BaseModelManager(
        feature_set="medium",
        targets_names_for_base_models=["target", "target_victor_20"],
        )


    predictors = model_manager.load_base_model_predictors()
    target_predictor = predictors["model_target"]
    target_victor_20_predictor = predictors["model_target_victor_20"]

    neutralized_predictors = model_manager.load_neutralized_base_model_predictors()
    neutralized_target_predictor = neutralized_predictors["neutralized_model_target"]
    neutralized_target_victor_20_predictor = neutralized_predictors["neutralized_model_target_victor_20"]

    

    predictor_weighted = meta_manager.load_predictor("weighted")
    predictor_lgbm = meta_manager.load_predictor("lgbm")

    # this will be end validation data that we will do scoring plots etc staff like that.
    # X = data_loader.load_validation_data() # this i only checked to see comparison with notebooks
    X = data_manager.load_validation_data_for_meta_model()

    # take top 2000 rows
    # X = X.head(2000)

    features = data_manager.get_features()
    preds_target = target_predictor(X)
    preds_target_neutralized = neutralized_target_predictor(X)
    preds_target_victor_20 = target_victor_20_predictor(X[features])
    preds_target_victor_20_neutralized = neutralized_target_victor_20_predictor(X[features])
    preds_weighted = predictor_weighted(X)
    preds_lgbm = predictor_lgbm(X)


    X["predictions_model_target"] = preds_target
    X["neutralized_predictions_model_target"] = preds_target_neutralized
    X["predictions_model_target_victor_20"] = preds_target_victor_20
    X["neutralized_predictions_model_target_victor_20"] = preds_target_victor_20_neutralized
    print(preds_weighted)
    print(X)
    X["meta_weighted_predictions"] = preds_weighted
    X["meta_lgbm_predictions"] = preds_lgbm

    scorer = Scorer()
    scores = scorer.compute_scores(X, "target")

    print(scores)

    print(X.head())

    # print(preds_weighted)
    # print(preds_lgbm)

if __name__ == "__main__":
    test_pipeline3()

