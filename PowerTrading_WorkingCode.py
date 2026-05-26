# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:48:22 2026

@author: jyoth
"""
'''
# CRISP-ML(Q):
Business & Data Understanding:
    
Business Problem: 
    
    In Power Trading,Price Volatility and demand-supply imbalances create significant 
  financial risk for market participants.The challenge lies in integrating multiple data sources,handling marketing 
  uncertainities and optimizing trading strategies in real-time.

Business objective: 
    
    Minimizing financial risks and procurement costs in power trading.

Business Constraints:
    
    Maximize Trading returns.

Success Criteria:
    
    Business Success Criteria: Atleast 15-20 % reduction in financial risks due to price volatility,leading to increased profitability for power traders.
    ML: Achieve a Mean Absolute Percentage Error (MAPE) below 10% compared to baseline models.
    Economic: Optimized trading strategies leading to 5 -10 % savings in energy procurement costs and a 10-15 % increase in returns from power trading.
        
HLD-LLD: 

Data Sources - Data Collection - Data Storage (DB, DWH, DL, DLH)

Data: The PowerTrading details are obtained from IEX india and is publicly available for us to access.        
      The Weather Details are obtained from Open-Meteo.com
      
Data Dictionary:
# - Dataset contains 112129 rows of  details for each 15 minutes timeblock
# - 12 features are recorded for each time block

Column Name	    |Column Description	          | Data Type	|Range	  |  Measures	Relevant (Y/N)
timestamp	    |Date and time of record	  |datetime	   2024–2026	date-time	Y
date	        |Trading date	              |date	       YYYY-MM-DD	date	    Y
block	        |15-min interval block (1–96) | int	       1–96     	index	    Y
hour	        |Hour of day	              |int	       0–23	        hours	    Y
day_of_week	    |Day of week (0=Mon)	      |int	       0–6	        day of week	Y
purchase_bid_MW	|Total demand bids	          |float	   0–50000	MW (Megawatts)	Y
sell_bid_MW	    |Total supply bids	          |float	   0–50000	MW (Megawatts)	Y
MCV_MW	        |Market Clearing Volume	      |float       0–50000	MW (Megawatts)	Y
volume_MWh	    |Energy traded in 15 min block|float	   0–50000	MWh (Megawatt-hour)	Y
MCP	            |Market Clearing Price	      |float	   0–20000	    Rs/MWh	    Y
temperature     |Air temperature	          |float	   -10 to 50	  °C	    Y
humidity        |Humidity	                  |float	   0–100	       %	    Y
wind_speed      |Wind speed	                  |float	   0–30       	m/s     	Y
cloud_cover     |Cloud cover     	          |float	   0–360	   degrees (0-360°)	N
solar_radiation |Solar radiation              |float	   0–1000	     W/m2	    Y

**Importing required packages**

'''
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
#-----------------------------------------------------------------------------

df.info()        

# datatypes of columns
df.head()
df.describe()        #EDA / Descriptive Statistics

#-----------1.Univariate ,Bivariate,bivariate(calculate covariant coefficients)

#df['MCP']=df["MCP (Rs/MWh) *"]
# Univariate
df["MCP (Rs/MWh) *"].describe()

# Skewness
df["MCP (Rs/MWh) *"].skew()

# Bivariate correlation
df[["MCP","Purchase Bid (MW)"]].corr()

# Covariance
df[["MCP","Purchase Bid (MW)"]].cov()

dtale.show(df)


#-----------3.Dimensionality Reduction using Simple National Mean
# Temperature
# Temperature national mean
temp_cols = df.filter(like="temperature").columns
df["national_temperature"] = df[temp_cols].mean(axis=1)

# Humidity national mean
humidity_cols = df.filter(like="humidity").columns
df["national_humidity"] = df[humidity_cols].mean(axis=1)

# Wind speed national mean
wind_cols = df.filter(like="wind_speed").columns
df["national_wind_speed"] = df[wind_cols].mean(axis=1)

# Cloud cover national mean
cloud_cols = df.filter(like="cloud_cover").columns
df["national_cloud_cover"] = df[cloud_cols].mean(axis=1)

# Solar radiation national mean
solar_cols = df.filter(like="shortwave_radiation").columns
df["national_solar_radiation"] = df[solar_cols].mean(axis=1)
print("Temperature columns:", temp_cols)
print("Humidity columns:", humidity_cols)
print("Wind columns:", wind_cols)
print("Cloud columns:", cloud_cols)
print("Solar columns:", solar_cols)
weather_city_cols = (
    list(temp_cols) +
    list(humidity_cols) +
    list(wind_cols) +
    list(cloud_cols) +
    list(solar_cols)
)
df.columns
nat = df
print
nat = nat.drop(columns=weather_city_cols)


weather_patterns = {
    "temperature_2m": "national_temperature",
    "relative_humidity_2m": "national_humidity",
    "wind_speed_10m": "national_wind_speed",
    "cloud_cover": "national_cloud_cover",
    "shortwave_radiation": "national_solar_radiation"
}

all_cols = []

for pattern, new_col in weather_patterns.items():
    cols = df.filter(like=pattern).columns
    df[new_col] = df[cols].mean(axis=1)
    all_cols.extend(cols)

national = df.drop(columns=all_cols)
# Save dataset
national.to_csv("national_weather_iex.csv",index=False)
df.to_csv("final_weather_iex.csv", index=False)
df_final = df
print("National means created successfully")
features = pd.read_csv(r"C:\Users\jyoth\PowerTrading/final_weather_iex.csv") 

import os
dtale.show(df,host='localhost',port=8000)

#-----------------4.Outlier Detection-------------------------------------

Q1 = df["MCP (Rs/MWh) *"].quantile(0.25)
Q3 = df["MCP (Rs/MWh) *"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df_final[(df_final["MCP (Rs/MWh) *"] < lower) | (df_final["MCP (Rs/MWh) *"] > upper)]

print(outliers)

print(outliers["MCP (Rs/MWh) *"])

national.columns

sns.boxplot(data=national[['national_temperature', 'national_humidity', 'national_wind_speed',
'national_solar_radiation']])

plt.show()

#------------------5.Calculate Business Moments -----------------------------
from scipy.stats import skew, kurtosis


features = [
'Purchase Bid (MW)',
'Sell Bid (MW)',
'MCV (MW)',
'Final Scheduled Volume (MW)',
'MCP (Rs/MWh)',
'national_temperature',
'national_humidity',
'national_cloud_cover',
'national_wind_speed',
'national_solar_radiation'
]

moments = pd.DataFrame({
    "Mean": df[features].mean(),
    "Median": df[features].median(),
    "Std Dev": df[features].std(),
    "Variance": df[features].var(),
    "Skewness": df[features].skew(),
    "Kurtosis": df[features].kurtosis()
})

print(moments)

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
#------------------------------------------------------------------------------
#------------             Data Preprocessing                   ----------------------------------------
#--------------------------------------------------------------------------------

'''

Handle Missing values
Remove Duplicates
Dimensionality reduction
Outlier treatment
Correlation analysis
Feature engineering-Create Time Features,lag features
Scaling'''

#------------------ 1.Handling Missing Values ------------------------------------
df.isnull().sum()
df.columns
#visualize missing values
df.isnull().sum().plot(kind='bar')
plt.title("Missing Values Count")
plt.show()



#------------------ 2.Remove Duplicates  ---------------------------------------

duplicate = features.duplicated()
sum(duplicate)

#------------------3.Feature Engineering-----------------------------------------
# Create Time Features
df_final = national

df_final.dtypes
df_final["Date"] = pd.to_datetime(df_final["Date"])
df_final["time"] = pd.to_datetime(df_final["time"])
df_final["day_of_week"] = df_final["Date"].dt.dayofweek
df_final["month"] = df_final["time"].dt.month
df_final["year"] = df_final["time"].dt.year
df_final["is_weekend"] = df_final["day_of_week"] >= 5



df_final["demand_supply_gap"] = df_final["Purchase Bid (MW)"] - df_final["Sell Bid (MW)"]

df_final["renewable_index"] = (
    df_final["national_wind_speed"] + df_final["national_solar_radiation"]
)

#lag Features

df_final['15mins_lag']=df_final['MCP (Rs/MWh) *'].shift(1)
df_final['30min_lag'] = df_final['MCP (Rs/MWh) *'].shift(2)
df_final['1hour_lag'] = df_final['MCP (Rs/MWh) *'].shift(4)
df_final['1day_lag'] = df_final['MCP (Rs/MWh) *'].shift(96)


lag_cols = ['15mins_lag','30min_lag','1hour_lag','1day_lag']

#df_final.drop(lag_cols)

#print(df_final.columns)
#------------------4.Outlier treatment ------------------------------------------
national.columns
sns.boxplot(data=national[["Purchase Bid (MW)","Sell Bid (MW)","MCV (MW)",'Final Scheduled Volume (MW)', 'MCP (Rs/MWh) *']])
plt.show()

sns.boxplot(data=features[['national_temperature', 'national_humidity', 'national_cover_cloud_',
'national_wind_speed']])
plt.show()
sns.boxplot(data=features[['national_solar_radiation']])
plt.show()

lower_cap = features["MCV (MW)"].quantile(0.05)
upper_cap = features["MCV (MW)"].quantile(0.95)

features["percentile MCV (MW)"] = features["MCV (MW)"].clip(lower_cap, upper_cap)

sns.boxplot(data=features[['percentile MCV (MW)']])
plt.show()

#from feature_engine.outliers import Winsorizer

#winsor_iqr = Winsorizer(capping_method='iqr',tail='end',fold =1.5,variables =[''])
features = df_final

market_cols = [
    "Purchase Bid (MW)","Sell Bid (MW)","MCV (MW)",'Final Scheduled Volume (MW)', 'MCP (Rs/MWh) *'
]
weather_cols =['national_temperature',
'national_wind_speed']

winsor_cols = market_cols + weather_cols
def winsorize_quantile(features,winsor_cols, lower_q=0.05, upper_q=0.95):
    for col in winsor_cols:
        lower = features[col].quantile(lower_q)
        upper = features[col].quantile(upper_q)
        features[col] = features[col].clip(lower, upper)
        
    return features
features = winsorize_quantile(features, winsor_cols)

sns.boxplot(data=features[["Purchase Bid (MW)","Sell Bid (MW)","MCV (MW)",'Final Scheduled Volume (MW)', 'MCP (Rs/MWh) *']])
plt.show()

sns.boxplot(data=features[['national_temperature', 'national_humidity', 'national_cover_cloud_',
'national_wind_speed']])
plt.show()
sns.boxplot(data=features[['national_solar_radiation']])
plt.show()

#-----------------------5.Log transformation---------------------------------------------------
#The density plot for features columns is skewed.It needs to be transformed
sns.kdeplot(df["national_solar_radiation"], label="national_solar_radiation")
sns.kdeplot(features[['national_temperature', 'national_humidity', 'national_cover_cloud_']], label="Weather variables")
sns.kdeplot(df[["Purchase Bid (MW)","Sell Bid (MW)","MCV (MW)",'Final Scheduled Volume (MW)', 'MCP (Rs/MWh) *']], label="Market Variables")

plt.legend()
plt.show()

import numpy as np
features.columns
cols = ['Purchase Bid (MW)', 'Sell Bid (MW)',
       'MCV (MW)', 'Final Scheduled Volume (MW)', 'MCP (Rs/MWh) *',
       'demand_supply_gap',  '15mins_lag',
       '30min_lag', '1hour_lag', '1day_lag'
]

for col in cols:
    df_final[col + "_log"] = np.log1p(df_final[col])   

#the above creates new columns MCP_log purchase_bid_MW_log,sell_bid_MW_log,MCV_MW_log,volume_MWh_log    







#-----------------------Scaling------------------------------------------------

num_cols = [
"Purchase Bid (MW)",
"Sell Bid (MW)",
"MCV (MW)",
'MCP (Rs/MWh) *',
"national_temperature",
'Final Scheduled Volume (MW)',
'national_humidity', 'national_cover_cloud_',
"national_wind_speed",
"demand_supply_gap"
]

df_final[num_cols].plot(kind="density")
plt.title("Density Plot of Market Variables")
plt.show()

df_final.columns
from sklearn.preprocessing import StandardScaler
df_final.columns

colu = ['Purchase Bid (MW)',
'Sell Bid (MW)', 'MCV (MW)', 'Final Scheduled Volume (MW)',
'MCP (Rs/MWh) *', 'national_temperature',
'national_humidity', 'national_wind_speed',
'national_solar_radiation', 'demand_supply_gap',
'renewable_index', '15mins_lag', '30min_lag',
'1hour_lag', '1day_lag']

df_final = df_final.drop(columns=colu)

scaler = StandardScaler()

numeric_cols = [col for col in df_final.columns if df_final[col].dtype in ["int64", "float64"]]
df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])
scaled_data = df_final

scaled_data.to_csv("scaled_iex_weather_data.csv")

#Plot for scaled 
cols = ["Purchase Bid (MW)","Sell Bid (MW)","MCV (MW)"]

df_final[num_cols].plot(kind="density")
plt.title("Density Plot of Market Variables")
plt.show()




#-------------------4.Correlation Analysis -------------------------------------
#------------MODEL BUILDING ---------------------------------------------------
#Data Preparation for Modeling
import pandas as pd
from sklearn.model_selection import train_test_split

scaled_data.columns

# Target
y = df_final["MCP (Rs/MWh) *_log"]

# Features
X = df_final.drop(columns=["MCP (Rs/MWh) *_log","Date","time",'Hour'])
df_final["Time Block"] = df_final["Time Block"].astype("category").cat.codes + 1
# Train test split (time series → no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
lr = LinearRegression()

lr.fit(X_train, y_train)

X.columns
y_pred = lr.predict(X_test)

print("MAE:", mean_absolute_error(y_test,y_pred))
print("R2:", r2_score(y_test,y_pred))


#XGBOOST 
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

print("MAE:", mean_absolute_error(y_test,y_pred))
print("R2:", r2_score(y_test,y_pred))


#------------To record the time taken ------------------------


# Target
y = df_final["MCP (Rs/MWh) *_log"]

# Features
X = df_final.drop(columns=["MCP (Rs/MWh) *_log","Date","time",'Hour'])

# Train/Test split (important for time series)
split = int(len(df_final)*0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

#Record Train/Test duration for documentation
train_start = df_final.iloc[:split]["Date"].min()
train_end = df_final.iloc[:split]["Date"].max()

test_start = df_final.iloc[split:]["Date"].min()
test_end = df_final.iloc[split:]["Date"].max()

print("Train Period:", train_start, "to", train_end)
print("Test Period:", test_start, "to", test_end)

#--------MAPE ----------------------------------

def MAPE(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100



mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

#-------------- Train Multiple Models ------------------------------------------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

train_pred_lr = lr.predict(X_train)
test_pred_lr = lr.predict(X_test)

train_mape_lr = MAPE(y_train,train_pred_lr)
test_mape_lr = MAPE(y_test,test_pred_lr)

#--Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200,random_state=42)

rf.fit(X_train,y_train)

train_pred_rf = rf.predict(X_train)
test_pred_rf = rf.predict(X_test)

train_mape_rf = MAPE(y_train,train_pred_rf)
test_mape_rf = MAPE(y_test,test_pred_rf)
print("train mape:",train_mape_rf)

#--XGBoost
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

xgb.fit(X_train,y_train)

train_pred_xgb = xgb.predict(X_train)
test_pred_xgb = xgb.predict(X_test)

train_mape_xgb = MAPE(y_train,train_pred_xgb)
test_mape_xgb = MAPE(y_test,test_pred_xgb)


#--96 Block Dayahead Forecast
forecast_96 = xgb.predict(X_test.iloc[:96])

forecast_df = pd.DataFrame({
    "Block":range(1,97),
    "Predicted_MCP":forecast_96
})

print(forecast_df)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)


#or directly
#model.score(X_test, y_test)


#----------------SAVE THE MODEL--------------------------------------------
#import joblib

#joblib.dump(model, "power_price_model.pkl")



#---------------------------------------------------------------------------
#------        1 Week Prediction                 ----------------------------
#-------------------------------------------------------------------------------
# 1 week prediction 
# price lag
df["price_lag_1day"] = df["MCP (Rs/MWh) *"].shift(96)
df["price_lag_2day"] = df["MCP (Rs/MWh) *"].shift(192)
df["price_lag_3day"] = df["MCP (Rs/MWh) *"].shift(288)
df["price_lag_7day"] = df["MCP (Rs/MWh) *"].shift(672)

# demand lag
df["demand_lag_1day"] = df["Final Scheduled Volume (MW)"].shift(96)
df["demand_lag_7day"] = df["Final Scheduled Volume (MW)"].shift(672)

# weather lag
df["temp_lag_1day"] = df["national_temperature"].shift(96)
df["solar_lag_1day"] = df["national_solar_radiation"].shift(96)










