import os
import pandas as pd
from src.preprocessing import convert_to_voltage, extract_features

def build_dataset(base_path='data_teh'):
    records = []

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

    return pd.DataFrame(records)