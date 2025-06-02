import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import pywt 


def convert_to_voltage(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df * (5.0 / 1023.0)

def zero_crossing_rate(data):
    if len(data) < 2:
        return 0
    return ((np.diff(np.sign(data)) != 0).sum()) / len(data)


def extract_features(df):
    features = {}
    for col in df.columns:
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(data) < 2:
            continue  # skip fitur yang terlalu sedikit

        # Statistik dasar
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_max'] = data.max()
        features[f'{col}_min'] = data.min()
        features[f'{col}_var'] = data.var()
        features[f'{col}_mode'] = data.mode().iloc[0] if not data.mode().empty else data.median()

        # Kuartil dan rentang
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        features[f'{col}_q1'] = q1
        features[f'{col}_med'] = data.median()
        features[f'{col}_q3'] = q3
        features[f'{col}_iqr'] = q3 - q1
        features[f'{col}_range'] = data.max() - data.min()

        # Skewness & kurtosis
        features[f'{col}_skew'] = skew(data, bias=False) if len(data) >= 3 else 0
        features[f'{col}_kurt'] = kurtosis(data, bias=False) if len(data) >= 3 else 0

        # Fitur sinyal
        features[f'{col}_zcr'] = zero_crossing_rate(data)

    return features
