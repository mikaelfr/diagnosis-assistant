import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def fix_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    continous = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age']
    categorical_assume_mean = ['sex']
    categorical_assume_no = ['motor_impairment', 'chest_pain']

    for col in continous:
        mean = df[col].mean()
        df[col] = df[col].fillna(mean)

    for col in categorical_assume_mean:
        mean = float(round(df[col].mean()))
        df[col] = df[col].fillna(mean)

    # it's better to assume no for some variables
    # assuming everyone has motor impairment just because
    # that wasn't written down is a big dangerous
    for col in categorical_assume_no:
        df[col] = df[col].fillna(0.0)

    return df

def convert_for_classifier(df: pd.DataFrame) -> np.array:
    samples = df.drop(['diagnosis'], axis=1)
    samples = fix_missing_values(samples)
    features = df['diagnosis']
    samples_arr = samples.to_numpy()
    features_arr = features.to_numpy()
    return (samples_arr, features_arr)

def classify_data(df: pd.DataFrame) -> RandomForestClassifier:
    X, y = convert_for_classifier(df)

    samples = convert_for_classifier(df)
    classfier = RandomForestClassifier()
    classfier.fit(X, y)
    return classfier
