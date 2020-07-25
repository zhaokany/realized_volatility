import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

PURCHASSED_PRICE = 1.5
FRESH_PRICE = 2.5
FROZEN_PRICE = 0.8
PENALTY = 1.0

N_COMPONENT_YEARLY = 6

def fresh_price(volume):
    return volume * FRESH_PRICE


def frozen_price(volume):
    return volume * FROZEN_PRICE


def loss_(y_obs, y_pred):
    """Compute the dealer gain

    :param float y_obs: real sales
    :param float y_pred: predicted sales = purchasses
    """
    expenses = y_pred * PURCHASSED_PRICE
    # if real sales are above the predicted ones
    # the only gain is the stock price, so y_pred
    if y_obs >= y_pred:
        return expenses + PENALTY - fresh_price(y_pred)
        # if real sales are below the predicted ones
    # we earn the fresh price for the sales of the day + frozen price of the leftover
    return expenses - (fresh_price(y_obs) + frozen_price(y_pred - y_obs))

loss = np.vectorize(loss_)

def trend_model(t, params):
    """Model

    :param np.array t: times for which we want to predict the optimum purchasses
    :param np.array params: model parameters
    """
    a, b = params
    return a * t + b


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


def seasonality_model(t, params):
    p = 52
    x = fourier_series(t, p, N_COMPONENT_YEARLY)
    return x @ params


def full_model(t, params):
    trend = trend_model(t, params[:2])
    yearly_seasonality = seasonality_model(t, params[2:N_COMPONENT_YEARLY*2+2])
    return trend + yearly_seasonality

def f(params, *args):
    """Function to minimize = losses for the dealer

    :param args: must contains in that order:
    - data to be fitted (pd.Series)
    - model (function)
    """
    data = args[0]
    model = args[1]
    t = data.t
    y_obs = data.y
    y_pred = model(t, params)
    gl = loss(y_obs, y_pred)
    l = gl.sum()
    return l

def demand_score(y_obs, y_pred):
    """Computes how often the offer was enough to satisfy the at least 90% of the demand
    """
    return ((y_obs - y_pred) / y_obs < 0.05).sum() / len(y_obs) * 100


data = pd.read_csv("../../../data/sample_data.csv")
data_train, data_test = data[:len(data)-52], data[len(data)-52:]


ARGS = (data_train, trend_model)
tol = 500
x0 = (-0.1, 75.0)
res_trend = minimize(f,
                              args=ARGS,
                              x0=x0,
                              tol=tol,
                              #options={"eps": 1e-10, "maxiter": 10000}
)

trend_param_init = res_trend.x.tolist()
yearly_param_init = [2.5, 2.5] * N_COMPONENT_YEARLY
param_init = trend_param_init + yearly_param_init

res = minimize(f, args=(data_train, full_model),
                        x0=param_init,
                        tol = 100,
                       )

if res.success:
    print("Fitted parameters  ;", res.x)
    print("Loss function value:", res.fun)
else:
    print(res.message)
    print(res.x)

params_fit = res.x

y_pred_train = full_model(data_train.index.values, res.x)
y_pred_test = full_model(data_test.index.values, res.x)
y_pred_trend = trend_model(data.index.values, res.x[:2])

plt.plot(data_train.t, y_pred_train, color="red", linewidth=3, label="Prediction on train sample")
plt.plot(data.t, data.y, 'k-', label="Observation")
plt.plot(data_test.t, y_pred_test, color="blue", linewidth=3, label="Prediction on test sample")
plt.plot(data.t, y_pred_trend, color="red", linewidth=2, linestyle="--", label="Prediction (trend)")
plt.xlabel("Date")
plt.ylabel("'A' quantity [Tons]")
plt.legend(loc="upper right")
plt.title("Trend modelling")
plt.show()

npr = demand_score(data_test.y, y_pred_test)

print(f"The dealer will be able to deliver at least 90% of the request in {npr:.1f}% of the cases")




