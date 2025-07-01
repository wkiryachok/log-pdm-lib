import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class KnnModel:
    def __init__(self, properties: dict = None):
        self.properties = properties
        self.model = None

    def fit(self, train_data: pd.DataFrame = None):
        self.model = NearestNeighbors(**self.properties).fit(train_data.values)

    def predict(self, test_data: pd.DataFrame = None) -> pd.Series:
        distances, _ = self.model.kneighbors(test_data.values)
        anomaly_score = pd.Series(distances.mean(axis=1), index=test_data.index)
        return anomaly_score
    

class EnvModel:
    def __init__(self, properties: dict = None, prototype_num: int = 10000):
        self.properties = properties
        self.prototype_num = prototype_num
        self.model = None

    def fit(self, train_data: pd.DataFrame = None):
        if len(train_data) > self.prototype_num:
            indexes = np.random.RandomState(42).choice(
                np.arange(len(train_data)),
                self.prototype_num,
                replace=False,
            )
            train_data = train_data.iloc[indexes].values
        else:
            train_data = train_data.values
        self.model = EllipticEnvelope(**self.properties).fit(train_data)

    def predict(self, test_data: pd.DataFrame = None) -> pd.Series:
        anomaly_score = pd.Series(
            np.sqrt(self.model.mahalanobis(test_data.values)),
            index=test_data.index,
        )
        return anomaly_score


class LofModel:
    def __init__(self, properties: dict = None):
        self.properties = properties
        self.model = None

    def fit(self, train_data: pd.DataFrame = None):
        self.model = LocalOutlierFactor(**self.properties).fit(train_data.values)

    def predict(self, test_data: pd.DataFrame = None) -> pd.Series:
        anomaly_score = pd.Series(
            self.model.score_samples(test_data.values),
            index=test_data.index,
        )
        return -anomaly_score


class IsoModel:
    def __init__(self, properties: dict = None):
        self.properties = properties
        self.model = None

    def fit(self, train_data: pd.DataFrame = None):
        self.model = IsolationForest(**self.properties).fit(train_data.values)

    def predict(self, test_data: pd.DataFrame = None) -> pd.Series:
        anomaly_score = pd.Series(
            self.model.score_samples(test_data.values),
            index=test_data.index,
        )
        return -anomaly_score


class SvmModel:
    def __init__(self, properties: dict = None, prototype_num: int = 10000):
        self.properties = properties
        self.prototype_num = prototype_num
        self.model = None

    def fit(self, train_data: pd.DataFrame = None):
        if len(train_data) > self.prototype_num:
            indexes = np.random.RandomState(42).choice(
                np.arange(len(train_data)),
                self.prototype_num,
                replace=False,
            )
            train_data = train_data.iloc[indexes].values
        else:
            train_data = train_data.values
        self.model = OneClassSVM(**self.properties).fit(train_data)

    def predict(self, test_data: pd.DataFrame = None) -> pd.Series:
        anomaly_score = pd.Series(
            self.model.score_samples(test_data.values),
            index=test_data.index,
        )
        return -anomaly_score + anomaly_score.max()
