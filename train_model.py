import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

def compare_models_with_random_search(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv"):
    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Pisahkan fitur dan label
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Tangani missing value
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    os.makedirs("output/figures", exist_ok=True)

    # Pipelines
    pipelines = {
        "Random Forest": Pipeline([
            ("var_filter", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=30)),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("var_filter", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=30)),
            ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("var_filter", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=30)),
            ("clf", xgb.XGBClassifier(eval_metric="mlogloss", random_state=42))
        ])
    }

    # Parameter spaces
    param_distributions = {
        "Random Forest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10]
        },
        "SVM": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto"]
        },
        "XGBoost": {
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__n_estimators": [100, 200, 300]
        }
    }

    results = []

    for name, pipeline in pipelines.items():
        print(f"\nTuning model: {name}")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions[name],
            n_iter=10,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train_enc)
        best_model = search.best_estimator_
        print(f"Best Params for {name}: {search.best_params_}")

        # Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        results.append({"Model": name, "Accuracy (%)": round(acc * 100, 2)})

        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test_enc, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"output/figures/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

    # Ringkasan hasil
    result_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False)
    print("\nPerbandingan Akurasi Model:\n")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    compare_models_with_random_search()
