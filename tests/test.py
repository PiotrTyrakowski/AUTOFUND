from numerai_automl.utils.utils import get_project_root

print(get_project_root())


from numerai_automl.data_managers.data_loader import DataLoader

data_loader = DataLoader(data_version="v5.0", feature_set="small")

data = data_loader.load_live_data()

print(data.columns)

print(data)
