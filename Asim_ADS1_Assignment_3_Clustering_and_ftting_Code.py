# Importing Modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')

# Loading dataset
df = pd.read_csv("API_19_DS2_en_csv_v2_4773766.csv", skiprows=4).iloc[:, :-1]
df.head()

# FIlling missing values
df.fillna(0, inplace=True)

# Dropping all world records
index = df[df["Country Name"] == "World"].index
df.drop(index, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# Extracting data of interested fields
cols = ['Population, total', "CO2 emissions (kt)"]
df_new = df[df["Indicator Name"].isin(cols)]

df_co2 = df_new[df_new["Indicator Name"] == "CO2 emissions (kt)"].drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1).set_index("Country Name").iloc[:, :-2]
df_co2.iloc[:, 30:].mean().plot(kind="line", figsize=(14, 5))
plt.title("Average CO2 Emission by the whole world from 1990 to 2019", fontsize=18)
plt.show()

temp = df_co2.iloc[:, 30:].mean(axis=1).sort_values(ascending=False)
temp = temp[temp > 0][9:29]
plt.figure(figsize=(16, 6))
sns.barplot(temp.index, temp.values)
plt.title("Top 20 Countries with Highest Emission of C02 from 1990 to 2019", fontsize=18)
plt.xticks(rotation=90)
plt.show()

# Selecting top interesting countries
top_countries = ["China", "North America", "United States","European Union", "India", "Pakistan"]

# Selecting data for only PK
df_pak = df_new[df_new["Country Name"] == "Pakistan"]

# Population of Pakistan from 1960 to 2021
df_pak.iloc[0, 4:].plot(kind="line", figsize=(13, 5))
plt.title("Population of Pakistan from 1960 to 2021")
plt.show()
