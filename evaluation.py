import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR #/ "data" / "processed"

def evaluate(test_preprocessed,run_id):

    X_test = test_preprocessed.drop("target", axis=1)
    y_test = test_preprocessed["target"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/voting-model")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
        
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_proba)
    
    with mlflow.start_run(run_id=run_id):    
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

    print(f"Evaluation completed | Recall = {recall:.3f}")

    return acc, precision, recall

if __name__ == "__main__":
    evaluate()