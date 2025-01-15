from matplotlib import pyplot as plt
from numerai_automl.data_managers import data_manager
from numerai_automl.ensemblers.weighted_ensembler import WeightedTargetEnsembler
from numerai_automl.model_managers.meta_model_manager import MetaModelManager
from numerai_automl.data_managers.data_manager import DataManager


def test_meta_model_manager():


    model_manager = MetaModelManager(
        targets_names_for_base_models=["target", "target_victor_20"],
        )

    data_manager = DataManager(data_version="v5.0", feature_set="small")    

    X = data_manager.load_live_data()
    print(X)


    
if __name__ == "__main__":
    test_meta_model_manager()


