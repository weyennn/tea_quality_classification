import os
import pandas as pd
from imblearn.over_sampling import ADASYN
from src.preprocessing import convert_to_voltage, extract_features

def build_dataset(base_path='data_teh', apply_adasyn=True):
    records = []

    # Loop semua file CSV dalam folder label
    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                df = convert_to_voltage(df)
                features = extract_features(df)
                features['label'] = label
                records.append(features)

    # Buat DataFrame awal
    dataset = pd.DataFrame(records)

    # Pisahkan fitur dan label
    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    # === [1] ADASYN ===
    if apply_adasyn:
        try:
            X, y = ADASYN().fit_resample(X, y)
        except Exception as e:
            print(f"[WARNING] ADASYN gagal: {e}")
            print("Melanjutkan tanpa balancing.")

    # Gabungkan kembali dengan label
    dataset_final = pd.DataFrame(X)
    dataset_final["label"] = y

    return dataset_final
