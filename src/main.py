"""
main.py

This is the main script to run a model given the specified yaml configuration file.
"""

from models.config import Config, get_model_instance
from dataloaders import get_data_splits
import mlflow
import mlflow.sklearn                    # NEW: for log_model
from mlflow.models import evaluate as ml_evaluate   # NEW: modern evaluate API
import pandas as pd                       # NEW: for DataFrame

if __name__ == "__main__":
    config = Config()

    # Now get the training and testing data splits
    train_dataset, test_dataset, tf_len = get_data_splits(config)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    # Depending on the architecture, we would instantiate different models
    model = get_model_instance(config, tf_len)

    # --- MLflow setup ---
    mlflow.set_experiment("tf-binding-PAX5")

    with mlflow.start_run(run_name=f"{config.architecture.value}_{config.tf}"):

        # log some params from the config
        mlflow.log_param("architecture", config.architecture.value)
        mlflow.log_param("tf", config.tf)
        mlflow.log_param(
            "pred_struct_features",
            [str(f) for f in config.pred_struct_features],
        )
        mlflow.log_param("use_probs", config.use_probs)
        mlflow.log_param("use_seq", getattr(config, "use_seq", True)) 

        # If you set any logreg hyperparams, log them too:
        mlflow.log_param("logreg_max_iter", 5000)
        mlflow.log_param("logreg_C", 1.0)

        # --- train + your own eval (prints + PNGs) ---
        model.train(train_dataset)
        metrics = model.evaluate(test_dataset)  # returns {"accuracy": ..., "auc": ...}

        # log your own metrics
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

        # log your own plots as artifacts, if they exist
        try:
            mlflow.log_artifact("logreg_pax5_confusion_matrix.png")
            mlflow.log_artifact("logreg_pax5_roc_curve.png")
        except FileNotFoundError:
            pass

        # --- MLflow built-in evaluation (MLflow 3.x style) ---
        # Only do this for models that provide get_arrays (your LogReg does)
        if hasattr(model, "get_arrays"):
            # 1) Prepare test data as a DataFrame
            X_test, y_test = model.get_arrays(test_dataset)
            df_test = pd.DataFrame(X_test)
            df_test["label"] = y_test

            # 2) Log the trained sklearn pipeline as an MLflow model
            model_info = mlflow.sklearn.log_model(
                sk_model=model.model,          # your sklearn pipeline
                artifact_path="model",         # will show up as 'model' in artifacts
            )

            # 3) Use mlflow.models.evaluate on the logged model URI
            eval_result = ml_evaluate(
                model=model_info.model_uri,    # <- this is the key difference
                data=df_test,
                targets="label",
                model_type="classifier",
                evaluators="default",          # default classifier evaluator
            )
            # You don't *have* to do anything with eval_result in code;
            # MLflow logs metrics + plots under the current run.
