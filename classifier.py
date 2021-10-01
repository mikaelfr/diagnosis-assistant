import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def convert_for_classifier(df: pd.DataFrame) -> np.array:
    samples = df.drop(['diagnosis'], axis=1)
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
