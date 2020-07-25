import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

PURCHASSED_PRICE = 1.5
FRESH_PRICE = 2.5
FROZEN_PRICE = 0.8


def fresh_price(volume):
    return volume * FRESH_PRICE


def frozen_price(volume):
    return volume * FROZEN_PRICE


def fourier_series(t, p=365.25, n=10):
    """
    :pram t: times
    :pram p: seasonality period. p=365.25 for yearly, p=7 for weekly seaonality
    :param n: number of terms in the fourrier serie
    """
    x = 2 * np.pi * np.arange(1, n + 1) / p
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


class TSEstimatory(BaseEstimator, RegressorMixin):

    def __init__(self, n_seasonal_components=6, **model_hyper_parameters):
        """
        """
        super().__init__()
        self.n_seasonal_components = n_seasonal_components
        # fitted parameters, initialized to None
        self.params_ = None

    # 1. Building the model
    @property
    def penalty(self):
        return 0

    def _seasonality_model(self, t, params):
        x = fourier_series(t, 52, self.n_seasonal_components)
        return x @ params

    def _model(self, t, params):
        trend = params[0] * t + params[1]
        seasonality = self._seasonality_model(t, params[2:self.n_seasonal_components*2+2])
        return trend + seasonality
        # return trend

    # 2. Define the loss function
    def _loss(self, y_obs, y_pred):
        """Compute the dealer gain

        :param np.array y_obs: real sales
        :param np.array y_pred: predicted sales = purchasses
        """
        # expenses = y_pred * PURCHASSED_PRICE
        # return np.where(
        #     y_obs >= y_pred,
        #     # if real sales are above the predicted ones
        #     # the only gain is the stock price, so y_pred
        #     expenses + self.penalty - fresh_price(y_pred),
        #     # if real sales are below the predicted ones
        #     # we earn the fresh price for the sales of the day + frozen price of the leftover
        #     expenses - (fresh_price(y_obs) + frozen_price(y_pred - y_obs))
        # ).sum()
        return sum((y_obs - y_pred)**2)

    # 3. Function to be minimized
    def _f(self, params, *args):
        """Function to minimize = losses for the dealer

        :param args: must contains in that order:
        - data to be fitted (pd.Series)
        - model (function)
        """
        data = self._train_data
        t = data.t
        y_obs = self._train_target
        y_pred = self._model(t.values, params)
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
        param_initial_values = [-0.1, 75.0] + [2.5, 2.5] * self.n_seasonal_components
        res = optimize.minimize(
            self._f,
            x0=param_initial_values,
            tol=100,
        )
        if res.success:
            self.params_ = res.x
        return self

    def predict(self, X):
        return self._model(X.t, self.params_)

    def demand_score(self, y_obs, y_pred):
        return sum((y_obs - y_pred)**2)

if __name__ == "__main__":
    e = TSEstimatory(n_seasonal_components=10)
    data = pd.read_csv("../../../data/sample_data.csv")
    data_train, data_test = data[:len(data) - 52], data[len(data) - 52:]

    # Cross validation
    scores = cross_val_score(e, data_train, data_train.y, cv=5, scoring="r2")
    print(scores)
    print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))

    # Grid search CV
    hyper_params = {
        "n_seasonal_components": np.arange(2, 20, 2)
    }
    clf = GridSearchCV(e, hyper_params, cv=5, scoring="r2")
    clf.fit(data_train, data_train.y)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_index_)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_pred = clf.predict(data_test)
    print()

    # y_pred_trend = trend_model(data.index.values, res.x[:2])

    e.fit(data_train, data_train.y)
    print(e.params_)
    y_pred_train = e._model(data_train.index.values, e.params_)
    y_pred_test = e._model(data_test.index.values, e.params_)
    loss_train = e._loss(y_pred_train, data_train.y)
    loss_test = e._loss(y_pred_test, data_test.y)
    print(f'loss_train = {loss_train}')
    print(f'loss_test = {loss_test}')

    plt.plot(data_train.t, y_pred_train, color="red", linewidth=3, label="Prediction on train sample")
    plt.plot(data.t, data.y, 'k-', label="Observation")
    plt.plot(data_test.t, y_pred, color="blue", linewidth=3, label="Prediction on test sample")
    # plt.plot(data.t, y_pred_trend, color="red", linewidth=2, linestyle="--", label="Prediction (trend)")
    plt.xlabel("Date")
    plt.ylabel("'A' quantity [Tons]")
    plt.legend(loc="upper right")
    plt.title("Trend modelling")
    # plt.show()

