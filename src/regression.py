#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


weather = pd.read_csv("weather-random.csv")


# In[4]:


weather.head(5)


# In[5]:


weather.drop(["(Inverters)","Random","Date"], axis = 1, inplace = True) 


# In[6]:


weather.shape


# In[7]:


weather.dtypes


# In[8]:


weather.head(7)


# In[9]:


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


# In[10]:


weather.columns


# In[11]:


X = np.array([CloudCoverage,Visibility,Temperature,DewPoint,RelativeHumidity,WindSpeed,StationPressure,Altimeter])
    
Y = np.array([SolarEnergy])


# In[12]:


# Model Intialization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg = LinearRegression()


# In[13]:


X.shape


# In[14]:


Y.shape


# In[15]:


X = X.transpose()


# In[16]:


X.shape


# In[17]:


Y = Y.transpose()


# In[18]:


# train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[19]:


# Data Fitting
reg = reg.fit(X_train,y_train)


# In[20]:


# Y Prediction
Y_pred = reg.predict(X_test)


# In[21]:


# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
r2 = reg.score(X, Y)


# In[22]:


print(rmse)
print(r2)


# In[23]:


from sklearn.externals import joblib


# In[24]:


joblib.dump(reg, 'reg.pkl')


# In[25]:


X_new= weather.loc[:, weather.columns != 'Solar energy']


# In[26]:


X_new.columns


# In[27]:


# Saving the data columns from training
model_columns = list(X_new.columns)
joblib.dump(model_columns, 'model_reg.pkl')
print("Models columns dumped!")


# In[28]:


# The coefficients
print('Coefficients: \n', reg.coef_)


# In[ ]: