import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load data
df_energy_consumption = pd.read_csv('energy_consumption.csv', sep=",", usecols=["start", "load"])
df_temp_radiation = pd.read_csv("temp_radiation.csv", sep=",")

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# DataFrame for Energy - consumption
df_energy_consumption = df_energy_consumption.rename(columns={"start": "time"})
df_energy_consumption["time"] = pd.to_datetime(df_energy_consumption["time"])
df_energy_consumption["hour"] = df_energy_consumption["time"].dt.hour
df_energy_consumption["date"] = df_energy_consumption["time"].dt.date
df_energy_consumption["month"] = df_energy_consumption["time"].dt.month
df_energy_consumption["year"] = df_energy_consumption["time"].dt.year
df_energy_consumption = df_energy_consumption.drop(df_energy_consumption.loc[df_energy_consumption["year"]==2020].index)

df_temp_radiation["time"] = pd.to_datetime(df_temp_radiation["utc_timestamp"])
df_temp_radiation_de = df_temp_radiation[["time", "DE_temperature", "DE_radiation_direct_horizontal", "DE_radiation_diffuse_horizontal"]]

df_energy_consumption = pd.merge(df_energy_consumption, df_temp_radiation_de, on="time")
#print(df_energy_consumption)




def consumption_yearly_average():
    df_energy_consumption_yearly = df_energy_consumption.groupby("year").mean()
    x = [2015, 2016, 2017, 2018, 2019]
    y1 = df_energy_consumption_yearly["load"]
    y2 = df_energy_consumption_yearly["DE_temperature"]
    y3 = df_energy_consumption_yearly["DE_radiation_direct_horizontal"]
    y4 = df_energy_consumption_yearly["DE_radiation_diffuse_horizontal"]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Year")
    ax1.plot(x, y2, label="Temperature (°C)", color="red")
    ax1.plot(x, y3, label="direct Radiation (W/m^2)", color="darkred")
    ax1.plot(x, y4, label="diffuse Radiation (W/m^2)", color="orange")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(x, y1, label="Energy consumption (MW)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.legend(loc=7)
    plt.title("yearly: energy and weather - Germany", loc="left")
    plt.xticks(x, x)

    plt.show()


consumption_yearly_average()




# Plot Energy consumption yearly average 
def consumption_daily_average():
    df_energy_consumption_yearly = df_energy_consumption.groupby("date").mean()
    x = df_energy_consumption_yearly.index
    y = df_energy_consumption_yearly["load"]

    plt.plot(x, y, label="Energy consumption (MW)")

    plt.legend()
    plt.title("yearly: energy and weather - Germany")

    plt.show()


consumption_daily_average()




# Plot: Energy consumption monthly average
def consumption_monthly_average():
    df_energy_consumption_monthly = df_energy_consumption.groupby("month").mean()
    x = np.arange(1, 13, 1)
    y1 = df_energy_consumption_monthly["load"]
    y2 = df_energy_consumption_monthly["DE_temperature"]
    y3 = df_energy_consumption_monthly["DE_radiation_direct_horizontal"]
    y4 = df_energy_consumption_monthly["DE_radiation_diffuse_horizontal"]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Month")
    ax1.plot(x, y2, label="Temperature (°C)", color="red")
    ax1.plot(x, y3, label="direct Radiation (W/m^2)", color="darkred")
    ax1.plot(x, y4, label="diffuse Radiation (W/m^2)", color="orange")
    ax1.set_xticklabels(months, rotation=40)
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(x, y1, label="Energy consumption (MW)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.legend(loc=1)
    plt.title("monthly: energy and weather - Germany", loc="left")
    plt.xticks(x, months, rotation="vertical")

    plt.show()


consumption_monthly_average()




# Plot: Energy consumption hourly average
def consumption_hourly_average():
    df_energy_consumption_hourly = df_energy_consumption.groupby("hour").mean()
    x = np.arange(0, 24, 1)
    y1 = df_energy_consumption_hourly["load"]
    y2 = df_energy_consumption_hourly["DE_temperature"]
    y3 = df_energy_consumption_hourly["DE_radiation_direct_horizontal"]
    y4 = df_energy_consumption_hourly["DE_radiation_diffuse_horizontal"]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("daily hour")
    ax1.plot(x, y2, label="Temperature (°C)", color="red")
    ax1.plot(x, y3, label="direct Radiation (W/m^2)", color="darkred")
    ax1.plot(x, y4, label="diffuse Radiation (W/m^2)", color="orange")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(x, y1, label="Energy consumption (MW)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.legend()
    plt.title("daily hours: energy and weather - Germany", loc="left")

    plt.show()


consumption_hourly_average()