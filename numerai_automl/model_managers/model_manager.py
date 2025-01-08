from typing import Dict, List
from config.config import TARGET_CANDIDATES
import os
import pandas as pd
from numerai_automl.model_trainer.model_trainer import ModelTrainer

# find folder models in my project
models_folder = os.path.join(os.path.dirname(__file__), "..", "models")
print(models_folder)


# TODO: I have no idea what should be correct relative path to models folder PLS HELP
class ModelManager():
    def __init__(self, targets: List[str], params: Dict):
        self.targets = targets
        print(TARGET_CANDIDATES)
    
    def train_models(self, features: pd.DataFrame, targets: pd.DataFrame):
        self.models = {}
        for target in self.targets:
            modelTrainer = ModelTrainer(self.params)
            modelTrainer.train(features, targets[target])
            model = modelTrainer.get_model()
            self.models[f"model_{target}"] = model
    
    def save_models(self):
        if self.models is None:
            raise Exception("Models do not exist")
        
        for target in self.targets:
            model_path = f"models/model_{target}.pkl"
            self.save_model(self.models[target], f"{target}.pkl")

    def load_models(self):
        pass
        
    