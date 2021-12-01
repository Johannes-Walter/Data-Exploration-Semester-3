import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import TimeSeriesSplit,cross_val_score, GridSearchCV


########## load data ##########
df_energy_consumption = pd.read_csv(r'energy_consumption/de.csv', sep=",", usecols=["start", "load"])


########## Data prep ##########
df_energy_consumption = df_energy_consumption.rename(columns={"start": "time", "load": "today_consumption"})

df_energy_consumption_daily = df_energy_consumption.copy()
df_energy_consumption_daily["time"] = pd.to_datetime(df_energy_consumption_daily["time"])
df_energy_consumption_daily["date"] = df_energy_consumption_daily["time"].dt.date
df_energy_consumption_daily = df_energy_consumption_daily.groupby("date").mean()

df_energy_consumption_daily.loc[:, "yesterday"] = df_energy_consumption_daily.loc[:, "today_consumption"].shift()
df_energy_consumption_daily.loc[:, "diff_to_yesterday"] = df_energy_consumption_daily.loc[:, "yesterday"].diff()

df_energy_consumption_daily = df_energy_consumption_daily.dropna()
#print(df_energy_consumption_daily)


########## training & test sets ##########
def split_data(data):
    date_sep = pd.to_datetime(20191231, format="%Y%m%d")
    x_train = data[:date_sep].drop(["today_consumption"], axis=1)
    y_train = data.loc[:date_sep, "today_consumption"]
    x_test = data[date_sep:].drop(["today_consumption"], axis=1)
    y_test = data.loc[date_sep:, "today_consumption"]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data(df_energy_consumption_daily)


########## try different Algorithms ##########
def compare_algo():
    algs = [('LR', LinearRegression()),
            ('EN', ElasticNet()),
            ('LA', Lasso()),
            ('RF', RandomForestRegressor(n_estimators=10)),
            ('ML', MLPRegressor(solver='lbfgs', max_iter=100000))]

    results = []
    names = []

    for name, model in algs:
        tscv = TimeSeriesSplit(n_splits=6)
        cv_results = cross_val_score(model, x_train, y_train, cv=tscv)
        results.append(cv_results)
        names.append(name)

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.xlabel("Alogrithm")
    plt.ylabel("cross-validation-score")
    plt.show()

compare_algo()


########## find Hyperparamter ##########
def find_best_model(xt, yt):
    model = RandomForestRegressor()
    param_search = {"max_features": ["auto", "sqrt", "log2"],
                    "max_depth": list(range(1, 10))}
    tscv = TimeSeriesSplit(n_splits=6)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search)

    gsearch.fit(xt, yt)
    best_model = gsearch.best_estimator_
    return best_model

#best_model = find_best_model(x_train, y_train)


def feature_importance():
    imp = best_model.feature_importances_
    features = x_train.columns
    indices = np.argsort(imp)
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), imp[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.show()

#feature_importance()


########## add Parameters for seasonality##########
# weekly (7 days)
df_energy_consumption_daily["last_week"] = df_energy_consumption_daily["today_consumption"].shift(7)
# montly (31 days)
df_energy_consumption_daily["last_month"] = df_energy_consumption_daily["today_consumption"].shift(31)

df_energy_consumption_daily = df_energy_consumption_daily.dropna()


x_train, y_train, x_test, y_test = split_data(df_energy_consumption_daily)
best_model = find_best_model(x_train, y_train)
feature_importance()


def plot_test():
    x = y_test.index.tolist()
    y1 = y_test.values
    y2 = best_model.predict(x_test)

    plt.plot(x, y1, color="green", label="actual")
    plt.plot(x, y2, color="red", label="predicted")

    plt.xticks(rotation="vertical")
    plt.ylabel("energy consumption (MW)")
    plt.title("Energy Consumption - Germany")
    plt.legend()
    plt.show()

plot_test()
