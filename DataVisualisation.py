# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:20:59 2026

@author: jyoth
"""

#----------------------------------------------------------------------------
#         Import the Required Libraries
#-----------------------------------------------------------------------------
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import numpy as np
#----------------------------------------------------------------------------
#         Load the Datasets
#-----------------------------------------------------------------------------
#

df = pd.read_excel(r"C:\Users\jyoth\Downloads\Dataset/IEX_Weather_final.xlsx")
#sql db
user = "root"
pw = 'jyothi'
db = 'power_db'

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df.to_sql('power_trading_cities_table',con = engine,if_exists ="replace")

sql = "select * from power_trading_cities_table;"
 
df = pd.read_sql_query(sql, engine)


#----------------------------------------------------------------------------
#         EDA   &  DATA CLEANING
#----------- box plot -----------------------------------------------------------

national = pd.read_csv(r"C:\Users\jyoth\PowerTrading/national_weather_iex.csv") 
features = national
sns.boxplot(data=national[['national_temperature', 'national_humidity', 'national_wind_speed',
'national_solar_radiation']])
#----------------------6.Distribution Analysis-----------------------------------
features.hist()
df["MCP (Rs/MWh) *"].hist()
df["Purchase Bid (MW)"].hist()
df["Sell Bid (MW)"].hist()


#Combined density plot
plt.rcParams["font.family"] = "DejaVu Sans"

sns.kdeplot(df["MCP (Rs/MWh) *"], label="MCP")
sns.kdeplot(features["national_solar_radiation"], label="national_solar_radiation")
sns.kdeplot(df["Final Scheduled Volume (MW)"], label="Final Scheduled Volume (MW)")

plt.legend()
plt.show()
print(features["national_solar_radiation"].describe())
print(features["national_solar_radiation"].nunique())
#--------------------6.Correlation Analysis------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

corr = national.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()


#Time Series Visualization

plt.figure(figsize=(12,5))
plt.plot(df_final['Date'], df_final['MCP (Rs/MWh) *'])
plt.title("Electricity Price Over Time")
plt.show()

#Daily Pattern

df_final.groupby("Time Block")["MCP (Rs/MWh) *"].mean().plot()
plt.title("Average MCP by 15-min Block")
plt.show()

#Day_of_Week 
df["day_of_week"] = df["Date"].dt.day_name()
df_final.groupby("day_of_week")["MCP (Rs/MWh) *"].mean().plot(kind='bar')
plt.title("MCP by Day of Week")
plt.show()

df_final.columns
#Weather vs Price Relationship
plt.scatter(df_final['national_temperature'], df_final['MCP_Rs_MWh'], alpha=0.3)
plt.xlabel("Temperature")
plt.ylabel("MCP")
plt.show()

#Solar radiation Impact
plt.scatter(df_final['national_solar_radiation'], df_final['MCP_Rs_MWh'], alpha=0.3)
plt.show()

df_final.columns