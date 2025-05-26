import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

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

def train_model_svm(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_svm_teh.pkl"):
    print(f"--- Training SVM Model ({model_path}) ---")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_path}' and '{test_path}' exist.")
        return

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV) untuk SVM...")
    svm_model = SVC(random_state=42, probability=True) # probability=True for future use if needed
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik (SVM): {grid_search.best_params_}")

    # Save best model and label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder_svm.pkl") # Save a specific encoder for SVM
    print(f"Model SVM disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report - SVM]")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (SVM + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")

def train_model_random_forest(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_rf_teh.pkl"):
    print(f"--- Training Random Forest Model ({model_path}) ---")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_path}' and '{test_path}' exist.")
        return

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV) untuk Random Forest...")
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik (Random Forest): {grid_search.best_params_}")

    # Save best model and label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder_rf.pkl") # Save a specific encoder for RF
    print(f"Model Random Forest disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report - Random Forest]")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (Random Forest + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")

def train_model_logistic_regression(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_lr_teh.pkl"):
    print(f"--- Training Logistic Regression Model ({model_path}) ---")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_path}' and '{test_path}' exist.")
        return

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'], # liblinear is good for small datasets, lbfgs for larger
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV) untuk Logistic Regression...")
    # Using 'liblinear' solver to allow 'l1' penalty
    lr_model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
    grid_search = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik (Logistic Regression): {grid_search.best_params_}")

    # Save best model and label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder_lr.pkl") # Save a specific encoder for LR
    print(f"Model Logistic Regression disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report - Logistic Regression]")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (Logistic Regression + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")

def train_model_knn(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_knn_teh.pkl"):
    print(f"--- Training K-Nearest Neighbors (KNN) Model ({model_path}) ---")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_path}' and '{test_path}' exist.")
        return

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV) untuk KNN...")
    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik (KNN): {grid_search.best_params_}")

    # Save best model and label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder_knn.pkl") # Save a specific encoder for KNN
    print(f"Model KNN disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report - KNN]")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (KNN + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")

    # Confusion Matrix

def train_model_lightgbm(train_path="output/train_dataset.csv", test_path="output/test_dataset.csv", model_path="output/model_lgbm_teh.pkl"):
    print(f"--- Training LightGBM Model ({model_path}) ---")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Ensure '{train_path}' and '{test_path}' exist.")
        return

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Encode label
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter grid for LightGBM
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [-1, 10, 20],
        'num_leaves': [31, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Grid search
    print("Melakukan pencarian hyperparameter (GridSearchCV) untuk LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_enc)

    best_model = grid_search.best_estimator_
    print(f"Model terbaik (LightGBM): {grid_search.best_params_}")

    # Save best model and label encoder
    joblib.dump(best_model, model_path)
    joblib.dump(le, "output/label_encoder_lgbm.pkl")
    print(f"Model LightGBM disimpan ke: {model_path}")

    # Prediksi
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluasi
    print("\n[Classification Report - LightGBM]")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (LightGBM + Hyperparameter Tuning)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    print("-" * 50 + "\n")


if __name__ == "__main__":
    # train_model_xgboost()
    train_model_random_forest()
    # train_model_logistic_regression()
    # train_model_knn()
    # train_model_svm()
    # train_model_lightgbm()