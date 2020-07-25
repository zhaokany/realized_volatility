import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels as sm
import scipy.stats as scs

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smapi.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smapi.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.graphics.gofplots.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return

def white_noise():
    np.random.seed(1)
    randser = np.random.normal(size=1000)
    print("Random Series\n -------------\nmean: %.3f \nvariance: %.3f \nstandard deviation: %.3f"
          % (randser.mean(), randser.var(), randser.std()))
    tsplot(randser, lags=30)

def random_walk():
    np.random.seed(1)
    n_samples = 1000

    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = x[t - 1] + w[t]
    tsplot(x, lags=30)
    # the first difference of a random walk is white noise
    # tsplot(np.diff(x), lags=30)

if __name__ == "__main__":
    random_walk()