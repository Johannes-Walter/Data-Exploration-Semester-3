import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose



########## load data ##########
df_energy_consumption = pd.read_csv(r"energy_consumption/de.csv", sep=",", usecols=["start", "load"])


########## Data prep ##########
df_energy_consumption = df_energy_consumption.rename(columns={"start": "time", "load": "today_consumption"})

df_energy_consumption_daily = df_energy_consumption.copy()
df_energy_consumption_daily["time"] = pd.to_datetime(df_energy_consumption_daily["time"])
df_energy_consumption_daily["date"] = df_energy_consumption_daily["time"].dt.date
df_energy_consumption_daily = df_energy_consumption_daily.groupby("date").mean()

test = seasonal_decompose(df_energy_consumption_daily,
                          period=365,
                          model="add")

test.plot()
print(test.resid, test.seasonal)