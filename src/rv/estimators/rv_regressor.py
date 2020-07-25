import numpy as np
from scipy import optimize
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class EWMAEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_lookback: int=5):
        """
        """
        super().__init__()
        assert n_lookback <= 50, "exponential moving is subject to be fewer than 50 days"
        self.n_lookback = n_lookback
        # fitted parameters, initialized to None
        self.params_ = None


    def _model(self, features, param):
        # weighted = features.iloc[:, self.n_lookback - 1]
        # for i in range(self.n_lookback - 2, -1, -1):
        #     weighted = param * weighted + (1.0 - param) * features.iloc[:, i]
        # return weighted
        try:
            features.iloc[:, 0:self.n_lookback] @ param
        except:
            a = 2
        return features.iloc[:, 0:self.n_lookback] @ param

    # 2. Define the loss function
    def _loss(self, y_obs, y_pred):
        try:
            summ = sum((y_obs - y_pred)**2)
        except:
            a = 2
        return sum((y_obs - y_pred)**2)

    # 3. Function to be minimized
    def _f(self, params, *args):
        """Function to minimize = losses for the dealer

        :param args: must contains in that order:
        - data to be fitted (pd.Series)
        - model (function)
        """
        data = self._train_data
        y_obs = self._train_target
        y_pred = self._model(data, params)
        l = self._loss(y_pred, y_obs)
        return l

    def fit(self, X, Y):
        """
        Fit global model on X features to minimize
        a given function on Y.

        @param X: train dataset (features, N-dim)
        @param Y: train dataset (target, 1-dim)
        """
        self._train_data = X
        self._train_target = Y
        param_initial_values = [1.0 / self.n_lookback] * self.n_lookback
        res = optimize.minimize(
            self._f,
            x0=param_initial_values,
            # bounds=[(0.0, 1.0)],
            tol=1e-6,
        )
        if res.success:
            self.params_ = res.x
        return self

    def predict(self, X):
        return self._model(X, self.params_)
