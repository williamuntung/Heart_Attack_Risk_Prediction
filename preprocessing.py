import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/Heart Attack Data Set.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    bool_cols = ['sex', 'fbs', 'exang']
    ordinal_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('ord', StandardScaler(), ordinal_cols),
            ('bool', 'passthrough', bool_cols)
        ]
    )
    
    preprocessor_train = pd.DataFrame(preprocessor.fit_transform(X_train)).reset_index(drop=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

    preprocessor_test= pd.DataFrame(preprocessor.transform(X_test)).reset_index(drop=True)  
    
    train_preprocessed = pd.concat([preprocessor_train.reset_index(drop=True),y_train.reset_index(drop=True)],axis=1)
    test_preprocessed = pd.concat([preprocessor_test.reset_index(drop=True),y_test.reset_index(drop=True)],axis=1)
    
    before_count = len(train_preprocessed)
    train_preprocessed = train_preprocessed.drop_duplicates()
    after_count = len(train_preprocessed)
    
    print(f"Total duplicates removed from train set: {before_count - after_count}")
    print("Total duplicates now in train set:", train_preprocessed.duplicated().sum())
    
    train_preprocessed.to_csv("artifacts/train_processed.csv", index=False)
    test_preprocessed.to_csv("artifacts/test_processed.csv", index=False)
    
    return train_preprocessed, test_preprocessed
    
if __name__ == "__main__":
    preprocess()