import Config 
import pandas as pd
from multiprocessing import Process
from timeit import timeit

from techniques.AnomaliesMethods import AnomaliesMethod
from techniques.RegressionMethods import RegressionMethod

def process_df():
    weather_data = pd.read_csv(Config.weather_path)
    generation_data = pd.read_csv(Config.generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])

def single_process_regression(rm):
    rm.adestrarMetodo(Config.mr['lr_pipe'],Config.mr['lr'])
    rm.adestrarMetodo(Config.mr['poly_pipe'],Config.mr['poly'])
    rm.adestrarMetodo(Config.mr['gBoosting_pipe'],Config.mr['gBoosting'])
    rm.adestrarMetodo(Config.mr['rf_pipe'],Config.mr['rf'])

def single_process_anomalies(am):
    am.isolationForest(Config.contamination)
    am.kmeans(Config.n_clusters)
    am.autoEncoder()

def multi_process_constructor(procesos):
    for proceso in procesos:
        proceso.start()    
    for proceso in procesos:
        proceso.join()

if __name__ == "__main__":
    
    tiempos = {}
    df = process_df()
    rm = RegressionMethod(df , Config.indepVars , Config.depVars)
    am = AnomaliesMethod(df)
    

    rm1 = Process(target= rm.adestrarMetodo, args=(Config.mr['lr_pipe'], Config.mr['lr']))
    rm2 = Process(target= rm.adestrarMetodo, args=(Config.mr['poly_pipe'], Config.mr['poly']))
    rm3 = Process(target= rm.adestrarMetodo, args=(Config.mr['gBoosting_pipe'], Config.mr['gBoosting']))
    rm4 = Process(target= rm.adestrarMetodo, args=(Config.mr['rf_pipe'], Config.mr['rf']))

    regression_process_list = [rm1,rm2,rm3,rm4]

    am1 = Process(target= am.isolationForest, args=(Config.contamination,))
    am2 = Process(target= am.kmeans, args=(Config.n_clusters,))
    am3 = Process(target= am.autoEncoder, args=())

    anomalies_process_list = [am1,am2,am3]

    #tiempos['single_process_regression'] = timeit("single_process_regression(rm)", globals=globals(), number=1)
    tiempos['single_process_anomalies'] = timeit("single_process_anomalies(am)", globals=globals(), number=1)

    #tiempos['regression_process_with_constructor'] = timeit("multi_process_constructor(regression_process_list)", globals=globals(), number=1)
    tiempos['anomalies_process_with_constructor'] = timeit("multi_process_constructor(anomalies_process_list)", globals=globals(), number=1)

    for metodo,tiempo in tiempos.items():
        print(f"Tiempo de ejecución para {metodo}: {tiempo/60:.2f} min")