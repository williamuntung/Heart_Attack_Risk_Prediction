import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

def train(train_preprocessed):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Streamlit-Pipeline")

    X_train = train_preprocessed.drop("target", axis=1)
    y_train = train_preprocessed["target"]

    with mlflow.start_run() as run:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=8
        )
        
        xgb = XGBClassifier(
            learning_rate = 0.001,
            max_depth = 5,
            n_estimators = 100
        )
        
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.005,
            max_depth=None,
            max_iter = 100,
            min_samples_leaf=15
        )
        
        model = VotingClassifier(
            estimators= [
                ('rf', rf),
                ('xgb', xgb),
                ('hgb', hgb)
            ],
            voting= 'soft',
            weights= [2, 1, 3]
        )

        model.fit(X_train, y_train)

        mlflow.log_params({
            "rf__n_estimators":100,
            "rf__max_depth" : 4,
            "rf__min_samples_split":8,
            "xgb__learning_rate":0.001,
            "xgb__max_depth":5,
            "xgb__n_estimators":100,
            "hgb__learning_rate":0.005,
            "hgb__max_depth":None,
            "hgb__max_iter":100,
            "hgb__min_samples_leaf":15
        })
        
        mlflow.sklearn.log_model(
            sk_model= model,
            artifact_path= "voting-model"
        )
        
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        
        return run.info.run_id



