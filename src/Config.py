import os
weather_path = os.path.join(os.path.dirname(__file__), "dataset", "Plant_1_Weather_Sensor_Data.csv")
generation_path = os.path.join(os.path.dirname(__file__), "dataset", "Plant_1_Generation_Data.csv")

depVars = "W"
indepVars = ["radiation", "temperature"]
n_clusters = 3