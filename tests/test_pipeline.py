from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager


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
    meta_manager.find_lgbm_ensemble()

    e

    ensemble_manager.create_predictions_by_ensemble()