import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

classfier_col_order = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age', 'sex', 'chest_pain', 'motor_impairment', 'diagnosis']

continous = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age']
categorical_assume_mean = ['sex']
categorical_assume_no = ['motor_impairment', 'chest_pain']
categorical = categorical_assume_mean + categorical_assume_no

def fix_missing_values(df: pd.DataFrame, means: dict = None) -> pd.DataFrame:
    """
    Can use means if we are using this in a ui and we cannot get means since we only have a single row
    """
    for col in continous:
        if means and col in means:
            mean = means[col]
        else:
            mean = df[col].mean()
        df[col] = df[col].fillna(mean)

    for col in categorical_assume_mean:
        if means and col in means:
            mean = means[col]
        else:
            mean = float(round(df[col].mean()))
        df[col] = df[col].fillna(mean)

    # it's better to assume no for some variables
    # assuming everyone has motor impairment just because
    # that wasn't written down is a bit dangerous
    for col in categorical_assume_no:
        df[col] = df[col].fillna(0.0)

    return df

def get_means_from_training_set(df: pd.DataFrame) -> dict:
    means = {}
    for col in continous:
        means[col] = df[col].mean()

    for col in categorical_assume_mean:
        means[col] = float(round(df[col].mean()))

    return means

def convert_for_classifier(df: pd.DataFrame) -> np.array:
    samples = df.drop(['diagnosis'], axis=1)
    samples = fix_missing_values(samples)
    features = df['diagnosis']
    samples_arr = samples.to_numpy()
    features_arr = features.to_numpy()
    return (samples_arr, features_arr)

def classify_data(df: pd.DataFrame) -> tuple:
    # Check correct column order for consistency
    if df.columns.tolist() != classfier_col_order:
        raise ValueError(f'Expected col order to be {classfier_col_order} but it was {df.columns.tolist()}')

    X, y = convert_for_classifier(df)

    classfier = RandomForestClassifier(bootstrap=True, criterion='gini', max_features=0.65, min_samples_leaf=20, min_samples_split=14, n_estimators=100)
    classfier.fit(X, y)

    means = get_means_from_training_set(df)
    return classfier, means
