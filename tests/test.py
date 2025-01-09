from numerai_automl.utils.utils import get_project_root

print(get_project_root())


from numerai_automl.data_managers.data_loader import DataLoader

data_loader = DataLoader()

print(data_loader.load_train_data())