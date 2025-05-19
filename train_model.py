import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model_xgboost(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_xgb_teh.pkl"):
    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV)...")
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik: {grid_search.best_params_}")

    # Save best model dan label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder.pkl")
    print(f"Model disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (XGBoost + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model_xgboost()
