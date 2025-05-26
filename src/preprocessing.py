import pandas as pd
from scipy.stats.mstats import skew, kurtosis


def convert_to_voltage(df):
    # Ubah semua nilai ke numerik, lalu hilangkan baris dengan NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df * (5.0 / 1023.0)

def extract_features(df):
    features = {}
    for col in df.columns:
        data = df[col]
        features[f'{col}_mean'] = data.mean()
        features[f'{col}_std'] = data.std()
        features[f'{col}_max'] = data.max()
        features[f'{col}_min'] = data.min()
        features[f'{col}_var'] = data.var()
        features[f'{col}_mode'] = data.mode()[0] if not data.mode().empty else None
        features[f'{col}_q1'] = data.quantile(0.25)
        features[f'{col}_med'] = data.median()
        features[f'{col}_q3'] = data.quantile(0.75)
        features[f'{col}_range'] = abs(data.max() - data.min())
        features[f'{col}_skew'] = skew(data)
        features[f'{col}_kurt'] = kurtosis(data)
    return features
