import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from rv.preprocessing import prepare, remove_overnight_jumps, calculate_intraday_fraction
from rv.calculator import RealizedVolatilityCalculator, VolatilitySignatureCalculator
from rv.estimators.rv_regressor import EWMAEstimator
from rv.utils.ts_plot import tsplot

if __name__ == "__main__":
    winsorized_limit = {
        "a": (0.05, 0.05),
        "b": (0.05, 0.05),
        "c": (0.05, 0.05),
        "d": (0.05, 0.05),
        "e": (0.05, 0.05),
        "f": (0.005, 0.003),
    }
    tenor_in_days = 22
    for stock_column_name in ["a", "b", "c", "d", "e", "f"]:
    # for stock_column_name in ["f"]:
        day_fraction_limits, intraday_limits = winsorized_limit[stock_column_name]
        print(stock_column_name)
        daily_data = prepare("../data/stockdata3.csv", stock_column_name)
        daily_data = remove_overnight_jumps(daily_data)
        intraday_fraction, overnight_fraction = calculate_intraday_fraction(daily_data, day_fraction_limits)
        print(f"intraday_fraction={intraday_fraction}")

        calculator = RealizedVolatilityCalculator
        rv = calculator.calculate_realized_volatilities(daily_data, intraday_fraction)
        print(rv[-1])

        # plt.plot(rv)
        # plt.show()

        #
        # n_hours_per_day = 9
        # n_minute_per_hour = 60
        #
        # n_days = 66
        # n_hours = n_days * n_hours_per_day
        #
        # daily_returns = prepare_daily(daily_data)
        # # tsplot(daily_returns['return'])
        # tau = 1.0 / 252
        # volatility_daily_frequency = np.sqrt(np.var(daily_returns['return']))
        # vsc_daily = VolatilitySignatureCalculator(n_days + 1)
        # volatility_signature_daily = vsc_daily.calculate(daily_returns, volatility_daily_frequency)
        # volatility_signature_daily_in_year = volatility_signature_daily / np.sqrt(tau)


        # hourly_returns = prepared_hourly(daily_data, 1.0 - intraday_fraction)[['return']]
        # tau = 1.0 / 8.5 / 252
        # volatility_hourly_frequency = np.sqrt(np.var(hourly_returns['return']))
        # print(f"volatility_hourly_frequency={volatility_hourly_frequency}")
        # tsplot(hourly_returns['return'])
        # vsc = VolatilitySignatureCalculator(n_hours + 1)
        # volatility_signature_hourly = vsc.calculate(hourly_returns, volatility_hourly_frequency)
        # volatility_signature_hourly_in_year = volatility_signature_hourly / np.sqrt(tau)

        # minute_returns = prepare_minute(daily_data, )[['return']]
        # # tsplot(minute_returns['return'])
        # volatility_minute_frequency = np.sqrt(np.var(minute_returns['return']))
        # vsc_minute = VolatilitySignatureCalculator(n_hours * n_minute_per_hour)
        # volatility_signature_minute = vsc_minute.calculate(minute_returns, volatility_minute_frequency)
        # tau_minute = 1.0 / 60.0 / 8.5 / 252
        # volatility_signature_minute_in_year = volatility_signature_minute / np.sqrt(tau_minute)
        # plt.plot(volatility_signature_minute_in_year, label="minute")
        # plt.plot(range(1, n_hours * n_minute_per_hour, n_minute_per_hour), volatility_signature_hourly_in_year, label="hourly")
        # plt.plot(range(1, n_days * n_hours * n_minute_per_hour, n_hours * n_minute_per_hour), volatility_signature_daily_in_year, label="daily")
        # plt.legend()
        # plt.show()


        # calculator = RealizedVolatilityCalculator(intraday_limits)
        # realized_vols = calculate_realized_vols(calculator, daily_data, intraday_fraction, tenor_in_days)
        # data = generate_rv_features(realized_vols, tenor_in_days)
        #
        # split_index = len(data) - tenor_in_days
        # data_train, data_test = data[:split_index], data[split_index:]
        # model = EWMAEstimator()
        #
        # # scores = cross_val_score(model, data_train.drop(columns=['y']), data_train.y, cv=5, scoring="r2")
        # # print(scores)
        # # print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
        #
        # # Grid search CV
        # hyper_params = {
        #     "n_lookback": np.arange(1, 6, 1)
        # }
        # clf = GridSearchCV(model, hyper_params, cv=5, scoring="r2")
        # clf.fit(data_train.drop(columns=['y']), data_train.y)
        #
        # print("Best parameters set found on development set:")
        # print()
        # print(clf.best_params_)
        # print()
        # print("Grid scores on development set:")
        # print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        # print()
        # print(clf.best_estimator_)
        # print(clf.best_params_)
        # print(clf.best_index_)
        #
        # print()
        # y_pred = clf.predict(data_test.drop(columns=['y']))
        # print(r2_score(data_test.y, y_pred))
