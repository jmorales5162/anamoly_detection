import os
import pandas as pd

datasetName = "dataset"
path = os.path.join(os.path.dirname(__file__), datasetName)



dt_p1_gen   = pd.read_csv(os.path.join(path ,'Plant_1_Generation_Data.csv'))
dt_p1_sens  = pd.read_csv(os.path.join(path ,'Plant_1_Weather_Sensor_Data.csv'))
dt_p2_gen   = pd.read_csv(os.path.join(path ,'Plant_2_Generation_Data.csv'))
dt_p2_sens  = pd.read_csv(os.path.join(path ,'Plant_2_Weather_Sensor_Data.csv'))

dt_p1_gen['DATE_TIME'] = pd.to_datetime(dt_p1_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
dt_p1_sens['DATE_TIME'] = pd.to_datetime(dt_p1_sens['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
dt_p2_gen['DATE_TIME'] = pd.to_datetime(dt_p2_gen['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
dt_p2_sens['DATE_TIME'] = pd.to_datetime(dt_p2_sens['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

#dt_source_key_specifico = dt_p1_gen.loc[dt_p1_gen['SOURCE_KEY'] == '1BY6WEcLGh8j5v7']

reduced_df = dt_p1_gen.groupby(['SOURCE_KEY', pd.Grouper(key='DATE_TIME', freq='15min')]).first().reset_index()
reduced_df.to_excel("ki.xlsx", index=False)