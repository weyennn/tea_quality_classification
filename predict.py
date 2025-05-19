import pandas as pd
import joblib
from src.preprocessing import convert_to_voltage, extract_features

def is_raw_sensor_data(df):
    return df.shape[0] > 100 and all(not any(stat in col for stat in ["_mean", "_std", "_skew", "_kurt"]) for col in df.columns)

def predict_teh(file_path, model_path="output/model_teh.pkl", reference_csv="/output/train_dataset.csv"):
    # Load model
    model = joblib.load(model_path)

    # Load input
    df = pd.read_csv(file_path)

    if is_raw_sensor_data(df):
        print("[INFO] Detected raw sensor data.")
        df = convert_to_voltage(df)
        features = extract_features(df)
        X_new = pd.DataFrame([features])
    else:
        print("[INFO] Detected extracted feature data.")
        X_new = df.copy()
        if "label" in X_new.columns:
            X_new = X_new.drop(columns=["label"])

    # Sinkronisasi kolom fitur
    ref_cols = pd.read_csv(reference_csv).drop(columns=["label"]).columns
    X_new = X_new.reindex(columns=ref_cols, fill_value=0.0)

    # Prediksi
    prediction = model.predict(X_new)[0]
    probs = model.predict_proba(X_new)[0]
    label_probs = dict(zip(model.classes_, probs))
    return prediction, label_probs

if __name__ == "__main__":
    file_path = "test_dataset.csv"  # atau data_teh/F1/F1_6.csv
    pred_label, prob_dict = predict_teh(file_path)

    print(f"\nPrediksi Jenis Teh: {pred_label}")
    print("Probabilitas Tiap Kelas:")
    for label, prob in prob_dict.items():
        print(f" - {label}: {prob:.4f}")
