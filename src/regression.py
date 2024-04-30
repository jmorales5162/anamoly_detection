import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

weather = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasetProba","weather-random.csv"))
weather.drop(["(Inverters)","Random","Date"], axis = 1, inplace = True) 

CloudCoverage = weather['Cloud coverage'].values
Visibility = weather['Visibility'].values
Temperature = weather['Temperature'].values
SolarEnergy= weather['Solar energy'].values

X = np.array([CloudCoverage,Visibility,Temperature])
Y = np.array([SolarEnergy])

reg = LinearRegression()

X = X.transpose(); Y = Y.transpose()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print("Shapes: X=" + str(X_train.shape) + " Y=" + str(y_train.shape))
reg = reg.fit(X_train,y_train)
Y_pred = reg.predict(X_test)
#print(Y_pred)

rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
r2 = reg.score(X, Y)


#print(y_test.shape)
#print(X_test[:,0].shape)
#plt.plot(X_test[:,0], Y_pred.flatten())
plt.scatter(X_test[:,0], Y_pred.flatten(), label="Prediction")
plt.scatter(X_test[:,0], y_test.flatten(), label="Real")
plt.xlabel('Cloud coverage'); plt.ylabel('production')
plt.title('Solar production')
plt.show()
#plt.plot(anomaly_test_data[0], 'b')
#plt.plot(decoder_out_a[0], 'r')
#plt.title("Model performance on Anomaly Data")
#plt.show()

#print(rmse)
#print(r2)

"""""
joblib.dump(reg, 'reg.pkl')
X_new= weather.loc[:, weather.columns != 'Solar energy']
X_new.columns

model_columns = list(X_new.columns)
joblib.dump(model_columns, 'model_reg.pkl')
print("Models columns dumped!")


print('Coefficients: \n', reg.coef_)
"""
