import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import os

weather = pd.read_csv("weather-random.csv")
weather.head(5)
weather.drop(["(Inverters)","Random","Date"], axis = 1, inplace = True) 
weather.shape
weather.dtypes
weather.head(7)

#Getting the variables to an array.
CloudCoverage = weather['Cloud coverage'].values
Visibility = weather['Visibility'].values
Temperature = weather['Temperature'].values
DewPoint= weather['Dew point'].values
RelativeHumidity= weather['Relative humidity'].values
WindSpeed= weather['Wind speed'].values
StationPressure= weather['Station pressure'].values
Altimeter= weather['Altimeter'].values
SolarEnergy= weather['Solar energy'].values


weather.columns

X = np.array([CloudCoverage,Visibility,Temperature,DewPoint,RelativeHumidity,WindSpeed,StationPressure,Altimeter])
Y = np.array([SolarEnergy])


reg = LinearRegression()


X.shape
Y.shape
X = X.transpose()
X.shape
Y = Y.transpose()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
reg = reg.fit(X_train,y_train)
Y_pred = reg.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
r2 = reg.score(X, Y)

print(rmse)
print(r2)


joblib.dump(reg, 'reg.pkl')
X_new= weather.loc[:, weather.columns != 'Solar energy']
X_new.columns

model_columns = list(X_new.columns)
joblib.dump(model_columns, 'model_reg.pkl')
print("Models columns dumped!")


print('Coefficients: \n', reg.coef_)

