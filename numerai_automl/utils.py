# numerai_automl/utils.py

import pandas as pd
import cloudpickle

def save_model(model, filename: str):
    with open(filename, "wb") as f:
        cloudpickle.dump(model, f)

def load_model(filename: str):
    with open(filename, "rb") as f:
        return cloudpickle.load(f)
