import os
weather_path = os.path.join(os.path.dirname(__file__), "dataset", "Plant_1_Weather_Sensor_Data.csv")
generation_path = os.path.join(os.path.dirname(__file__), "dataset", "Plant_1_Generation_Data.csv")

depVars = ['DC_POWER']
indepVars = ['IRRADIATION', 'MODULE_TEMPERATURE']

n_clusters = 3