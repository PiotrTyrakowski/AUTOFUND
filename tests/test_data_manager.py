
from numerai_automl.data_managers.data_manager import DataManager


def test_data_manager():
    data_manager = DataManager()

    data = data_manager.load_validation_data_for_neutralization_of_base_models()

    print(data.columns)
    print(data)

if __name__ == "__main__":
    test_data_manager()