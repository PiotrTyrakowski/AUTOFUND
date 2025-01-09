# numerai_automl/automl_pipeline.py

import pandas as pd
from numerai_automl.data_loader import DataLoader
from numerai_automl.model_trainer import ModelTrainer
from numerai_automl.feature_neutralizer import FeatureNeutralizer
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.reporter import Reporter
from numerai_automl.utils.utils import save_model
import cloudpickle
from numerai_automl.config import LIGHTGBM_PARAMS_OPTION, FEATURE_SET_OPTION, TARGET_CANDIDATES

def main():
    # Initialize DataLoader
    data_loader = DataLoader(data_version="v5.0", download_data=False) # change this to False if you already have the data

    targets_list = TARGET_CANDIDATES
    lightgbm_params = LIGHTGBM_PARAMS_OPTION
    feature_set = FEATURE_SET_OPTION
    
    # Load data
    train = data_loader.load_train_data(feature_set=feature_set, target_set=targets_list, downsample_step=4)
    validation = data_loader.load_validation_data(feature_set=feature_set, target_set=targets_list, downsample_step=4,
                                                  start_index=1)
    
    # Extract features and target
    feature_cols = train.columns.tolist()
    feature_cols.remove("era")
    for target in targets_list:
        feature_cols.remove(target)

    X_train = train[feature_cols]
    y_train_list = train[targets_list]
    
    X_val = validation[feature_cols]
    y_val = validation[["target"]]

    # Initialize and train models with different parameter options
    trainer = ModelTrainer(params=lightgbm_params)


    models = {}
    for i, target in enumerate(targets_list, start=1):
        model_path = f"models/model_{target}.pkl"
        
        # Check if model exists
        try:
            with open(model_path, 'rb') as f:
                model = cloudpickle.load(f)
                print(f"Loading existing model for {target}")
        except (FileNotFoundError, EOFError):
            print(f"Training new model for {target}")
            model = trainer.train(X_train, y_train_list[target])
            save_model(model, model_path)
            
        models[f"model_{target}"] = model
    
    # Feature Neutralization
    # neutralizer = FeatureNeutralizer(neutralizer_features=feature_cols, proportion=1.0)
    predictions = {}
    for name, model in models.items():
        predictions_path = f"predictions/{name}_preds.pkl"
        
        # Check if predictions exist
        try:
            with open(predictions_path, 'rb') as f:
                model_predictions = cloudpickle.load(f)
                print(f"Loading existing predictions for {name}")
                predictions[name] = model_predictions
        except (FileNotFoundError, EOFError):
            print(f"Generating new predictions for {name}")
            predictions[name] = model.predict(X_val)
            
            with open(predictions_path, 'wb') as f:
                cloudpickle.dump(predictions[name], f)
    
    predictions = pd.DataFrame(predictions)
    # neutralized_predictions = neutralizer.apply(predictions, X_val)()
    print("after predictions")

    print(predictions["model_target"])
    print(y_val["target"])

    exit()

    # TODO: i will need to repair this.
    
    # Scoring
    scorer = Scorer()
    corr_scores = scorer.compute_corr(predictions["model_target"], y_val["target"])


    meta_model = pd.Series() 
    mmc_scores = scorer.compute_mmc(predictions, meta_model, y_val)
    
    # Combine scores
    scores = pd.concat([corr_scores, mmc_scores], axis=1)
    scores.columns = ["CORR", "MMC"]
    
    # Reporting
    reporter = Reporter(scores=scores)
    summary = reporter.generate_summary()
    print(summary)
    reporter.plot_cumulative_scores()
    reporter.plot_score_distribution()
    reporter.save_report("numerai_automl_report.pdf")
    
    # Serialize the entire pipeline if needed
    # pipeline = {
    #     "models": models,
    #     "neutralizer": neutralizer,
    #     "scorer": scorer,
    #     "scores": scores
    # }
    # with open("automl_pipeline.pkl", "wb") as f:
    #     cloudpickle.dump(pipeline, f)

if __name__ == "__main__":
    main()
