"""Utilities for explanation generator."""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def missingdata_handler(X):
    si_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    si_mean.fit(X)
    return pd.DataFrame(si_mean.transform(X), columns=X.columns)


def normalizer(X):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)