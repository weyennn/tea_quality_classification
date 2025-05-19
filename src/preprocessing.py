import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def convert_to_voltage(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()  # hilangkan baris yang ada NaN
    return df * (5.0 / 1023.0)


def extract_features(df):
    features = {}
    for col in df.columns:
        data = df[col]
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_max'] = data.max()
        features[f'{col}_min'] = data.min()
        features[f'{col}_skew'] = skew(data)
        features[f'{col}_kurt'] = kurtosis(data)
    return features
