# numerai_automl/utils.py

import pandas as pd
import cloudpickle
import os

def save_model(model, filename: str):
    with open(filename, "wb") as f:
        cloudpickle.dump(model, f)

def load_model(filename: str):
    with open(filename, "rb") as f:
        return cloudpickle.load(f)
import os


def get_project_root():
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

