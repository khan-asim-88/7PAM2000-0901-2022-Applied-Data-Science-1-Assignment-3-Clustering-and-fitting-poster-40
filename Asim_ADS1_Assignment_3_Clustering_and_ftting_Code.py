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

# CO2 Emission of Pakistan from 1960 to 2021
df_pak.iloc[1, 4:].plot(kind="line", figsize=(13, 5))
plt.title("CO2 Emission of Pakistan from 1960 to 2021")
plt.show()

temp = df_pak.iloc[1, -32:].sum()/df_co2.sum(axis=1).sum()

print(f"Contribution of Pakistan in CO2 emission is {round(temp, 5)} percent.")

# Change in CO2 Emission of Selected Countries from 1990 to 2019
df_temp = df_new[df_new["Country Name"].isin(top_countries)]
df_temp = df_temp[df_temp["Indicator Name"] == "CO2 emissions (kt)"]
df_temp.set_index("Country Name").iloc[:, 33:-2].plot(kind="line", figsize=(15, 6), legend=False)
plt.title("CO2 Emission of Selected Countries from 1990 to 2019")
plt.show()

# CO2 Emission of Selected Countries from 1990 to 2019
df_temp.set_index("Country Name").iloc[:, 33:-2].T.plot(kind="line", figsize=(15, 6), legend=True)
plt.title("CO2 Emission of Selected Countries from 1990 to 2019")
plt.show()

# Preparing data for Clustering
df_new = df_new[df_new["Indicator Name"] == "CO2 emissions (kt)"]

count_mapper = dict([tuple([i, j]) for i, j in zip(df_new["Country Name"].unique(), range(df_new["Country Name"].nunique()))])
df_new["Country Name"] = df_new["Country Name"].map(count_mapper)
df_new.drop(df_new.select_dtypes("O").columns, axis=1, inplace=True)

# Transforming data to 2D for plotting
X = df_new.copy()
from sklearn.decomposition import PCA
X_new = PCA(2).fit_transform(X)

df_new[["X1", "X2"]] = X_new

# Clustering data
from sklearn.cluster import KMeans
kmean = KMeans(3)
kmean.fit(X)
y_pred = kmean.predict(X)

df_new["cluster"] = y_pred
df_new["Country Name"] = df_new["Country Name"].map({v: k for k, v in count_mapper.items()})
df_new["AVG_CO2"] = df_new.iloc[:, 1:-1].mean(axis=1)
