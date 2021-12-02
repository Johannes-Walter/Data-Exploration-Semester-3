import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV

from sklearn.metrics import explained_variance_score


########## load data ##########
df_energy_consumption = pd.read_csv('energy_consumption.csv', sep=",", usecols=["start", "load"])                       # Energy Verbrauch aus Deutschland
df_temp_radiation = pd.read_csv("temp_radiation.csv", sep=",", usecols=["utc_timestamp",
                                                                        "DE_temperature",
                                                                        "DE_radiation_direct_horizontal",
                                                                        "DE_radiation_diffuse_horizontal"])             # Temperatur und Radiation Daten aus Deutschland


########## Data prep ##########
df_energy_consumption = df_energy_consumption.rename(columns={"start": "time",
                                                              "load": "today_consumption"})                             # Spalten umbenennen

df_temp_radiation = df_temp_radiation.rename(columns={"utc_timestamp": "time",
                                                      "DE_temperature": "temperature",
                                                      "DE_radiation_direct_horizontal": "radiation_direct",
                                                      "DE_radiation_diffuse_horizontal": "radiation_diffuse"})          # Spalten umbenennen

df_energy_consumption["time"] = pd.to_datetime(df_energy_consumption["time"])                                           # Zeit in datetime Format
df_temp_radiation["time"] = pd.to_datetime(df_temp_radiation["time"])

df_energy_consumption_daily = df_energy_consumption.copy()
df_energy_consumption_daily["date"] = df_energy_consumption_daily["time"].dt.date                                       # Datumsspalte ohne Zeit
df_energy_consumption_daily = df_energy_consumption_daily.groupby("date").sum()                                         # Verbrauch pro Tag summieren

df_energy_consumption_daily.loc[:, "yesterday"] = df_energy_consumption_daily.loc[:, "today_consumption"].shift()       # Spalte mit Verbrauch vom Vortag
df_energy_consumption_daily.loc[:, "diff_to_yesterday"] = df_energy_consumption_daily.loc[:, "yesterday"].diff()        # Spalte mit Differenz zum Vortag

df_energy_consumption_daily = df_energy_consumption_daily.dropna()                                                      # NAs entfernen
#print(df_energy_consumption_daily)


########## training & test sets ##########
def split_data(data):
    x = data.drop(["today_consumption"], axis=1)
    y = data["today_consumption"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)                                            # Test und Trainingsdaten trennen
    return x_train, y_train, x_test, y_test

#x_train, y_train, x_test, y_test = split_data(df_energy_consumption_daily)


########## try different Algorithms ##########
def compare_algo():
    algs = [('LR', LinearRegression()),
            ('EN', ElasticNet()),
            ('LA', Lasso()),
            ('RF', RandomForestRegressor(n_estimators=10))]                                                             # verschieden Alogrithmen angeben

    results = []
    names = []

    for name, model in algs:
        tscv = TimeSeriesSplit(n_splits=6)
        cv_results = cross_val_score(model, x_train, y_train, cv=tscv)
        results.append(cv_results)
        names.append(name)                                                                                              # Cross Validation für alle Algorithmen durchführen

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.xlabel("Alogrithm")
    plt.ylabel("cross-validation-score")
    plt.show()                                                                                                          # Cross Validation Scores plotten

#compare_algo()


########## find Hyperparamter ##########
def find_best_model(xt, yt):
    model = RandomForestRegressor()
    param_search = {"max_features": ["auto", "sqrt", "log2"],
                    "max_depth": list(range(1, 10))}
    tscv = TimeSeriesSplit(n_splits=6)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search)

    gsearch.fit(xt, yt)
    best_model = gsearch.best_estimator_
    return best_model                                                                                                   # verschiedene Parameter für bestes Modell testen

#best_model = find_best_model(x_train, y_train)


def feature_importance():
    imp = best_model.feature_importances_
    features = x_train.columns
    indices = np.argsort(imp)                                                                                           # Features auf Wichtigkeit testen
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), imp[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.show()                                                                                                          # Feature Wichtigkeit plotten

#feature_importance()


def plot_test():
    d = {"date": y_test.index.tolist(),
         "actual": y_test.values,
         "predicted": best_model.predict(x_test)}                                                                       # mit Modell prediction auf Test-Daten
    df_plot = pd.DataFrame(data=d)
    df_plot = df_plot.sort_values(by=["date"])

    x = df_plot["date"]
    y1 = df_plot["actual"]
    y2 = df_plot["predicted"]

    plt.plot(x, y1, color="green", label="actual")
    plt.plot(x, y2, color="red", label="predicted")

    plt.xticks(rotation="vertical")
    plt.ylabel("energy consumption (MW)")
    plt.title("Energy Consumption - Germany")
    plt.legend()
    plt.show()                                                                                                          # Tatsächliche und predicted Daten plotten

#plot_test()
#print(explained_variance_score(y_test.values, best_model.predict(x_test)))




########## finallay prediction with more features and hourly data ##########
# hourly Data Frame
df_energy_consumption_hourly = df_energy_consumption.copy()
df_energy_consumption_hourly["time"] = df_energy_consumption_hourly["time"].dt.floor("H")                               # Minuten Angaben entfernen
df_energy_consumption_hourly = df_energy_consumption_hourly.groupby("time").sum()                                       # Verbrauch pro Stunde summieren

#add Yesterday column
df_energy_consumption_hourly.loc[:, "yesterday"] = df_energy_consumption_hourly.loc[:, "today_consumption"].shift(24)   # Wert 24h zurück als Spalte
df_energy_consumption_hourly.loc[:, "diff_to_yesterday"] = df_energy_consumption_hourly.loc[:, "yesterday"].diff()      # Differenz zum Vortag

#add Parameters for seasonality
# weekly (7 days)
df_energy_consumption_hourly["last_week"] = df_energy_consumption_hourly["today_consumption"].shift(7*24)               # Wert der vorherigen Wochen als Spalte
# montly (31 days)
df_energy_consumption_hourly["last_month"] = df_energy_consumption_hourly["today_consumption"].shift(31*24)             # Wert des vorherigen Monats als Spalte
# yearly (365 days)
df_energy_consumption_hourly["last_year"] = df_energy_consumption_hourly["today_consumption"].shift(365*24)             # Wert des vorherigen Jahres als Spalte

# add weather data
df_energy_consumption_hourly = pd.merge(df_energy_consumption_hourly, df_temp_radiation, on="time")                     # Wetter Daten hinzufügen

df_energy_consumption_hourly = df_energy_consumption_hourly.dropna()                                                    # NAs entfernen
df_energy_consumption_hourly = df_energy_consumption_hourly.set_index("time")                                           # Zeit Spalte als Index


x_train, y_train, x_test, y_test = split_data(df_energy_consumption_hourly)
best_model = find_best_model(x_train, y_train)
feature_importance()
print(explained_variance_score(y_test.values, best_model.predict(x_test)))
#plot_test()