import Config 
import pandas as pd
from multiprocessing import Process
from timeit import timeit
from pathos.multiprocessing import ProcessingPool

from techniques.AnomaliesMethods import AnomaliesMethod
from techniques.RegressionMethods import RegressionMethod

def process_df():
    """Preprocesamiento de los datos

    - Se leen los datos del csv
    - Se preprocesan los datos para su posterior análisis"""
    weather_data = pd.read_csv(Config.weather_path)
    generation_data = pd.read_csv(Config.generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])

def create_process():
    """Crea las listas de procesos para los modelos de regresión y de detección de anomalías"""
    rm1 = Process(target= rm.adestrarMetodo, args=(Config.mr['lr_pipe'], Config.mr['lr']))
    rm2 = Process(target= rm.adestrarMetodo, args=(Config.mr['poly_pipe'], Config.mr['poly']))
    rm3 = Process(target= rm.adestrarMetodo, args=(Config.mr['gBoosting_pipe'], Config.mr['gBoosting']))
    rm4 = Process(target= rm.adestrarMetodo, args=(Config.mr['rf_pipe'], Config.mr['rf']))
    process_list = [rm1,rm2,rm3,rm4]

    am1 = Process(target= am.isolationForest, args=(Config.contamination,))
    am2 = Process(target= am.kmeans, args=(Config.n_clusters,))
    am3 = Process(target= am.autoEncoder, args=())
    anomalies_process = [am1,am2,am3]
  
    return process_list, anomalies_process 

def single_process_anomalies(am):
    """Se crean los procesos de detección de anomalías secuenciales de la clase Process

    :param am: Objeto de la clase AnomaliesMethod.
    :type am: object"""
    am.isolationForest(Config.contamination)
    am.kmeans(Config.n_clusters)
    am.autoEncoder()

def single_process_regression(rm):
    """Se crean los procesos de regresion secuenciales de la clase Process

    :param rm: Objeto de la clase RegressionMethod.
    :type rm: object"""
    rm.adestrarMetodo(Config.mr['lr_pipe'],Config.mr['lr'])
    rm.adestrarMetodo(Config.mr['poly_pipe'],Config.mr['poly'])
    rm.adestrarMetodo(Config.mr['gBoosting_pipe'],Config.mr['gBoosting'])
    rm.adestrarMetodo(Config.mr['rf_pipe'],Config.mr['rf'])

def multi_process_constructor(procesos):
    """Ejecuta una lista de procesos

    :param procesos: Lista de procesos a ejecutar
    :type procesos: list"""
    for proceso in procesos:
        proceso.start()    
    for proceso in procesos:
        proceso.join()

def anomalies_process_with_pathos():
    """Ejecuta una lista de procesos de detección de anomalías

    :param procesos: Lista de procesos a ejecutar
    :type procesos: list"""
    with ProcessingPool() as pool:
        rm_1 = pool.apipe(am.isolationForest, Config.contamination)
        rm_2 = pool.apipe(am.kmeans, Config.n_clusters)
        rm_3 = pool.apipe(am.autoEncoder)
    procesos = [rm_1,rm_2,rm_3] 
    for proceso in procesos:
        proceso.get()

def regression_process_with_pathos():
    """Ejecuta una lista de procesos de regresion

    :param procesos: Lista de procesos a ejecutar
    :type procesos: list"""
    with ProcessingPool() as pool:
        rm_1 = pool.apipe(rm.adestrarMetodo, Config.mr['lr_pipe'], Config.mr['lr'])
        rm_2 = pool.apipe(rm.adestrarMetodo, Config.mr['poly_pipe'], Config.mr['poly'])
        rm_3 = pool.apipe(rm.adestrarMetodo, Config.mr['gBoosting_pipe'], Config.mr['gBoosting'])
        rm_4 = pool.apipe(rm.adestrarMetodo, Config.mr['rf_pipe'], Config.mr['rf'])
    procesos = [rm_1,rm_2,rm_3,rm_4] 
    for proceso in procesos:
        proceso.get()

if __name__ == "__main__":
    
    tiempos = {}
    df = process_df()
    rm = RegressionMethod(df , Config.indepVars , Config.depVars)
    am = AnomaliesMethod(df)
    
    process_list , anomalies_process = create_process()

    tiempos['single_process_anomalies'] = timeit("single_process_anomalies(am)", globals=globals(), number=1)
    tiempos['anomalies_process_with_constructor'] = timeit("multi_process_constructor(anomalies_process)", globals=globals(), number=1)
    tiempos['anomalies_process_with_pathos'] = timeit("anomalies_process_with_pathos()", globals=globals(), number=1) 

    tiempos['single_process_regression'] = timeit("single_process_regression(rm)", globals=globals(), number=1)
    tiempos['regression_process_with_constructor'] = timeit("multi_process_constructor(process_list)", globals=globals(), number=1)
    tiempos['regression_process_with_pathos'] = timeit("regression_process_with_pathos()", globals=globals(), number=1)
    
    datos = pd.DataFrame(tiempos, index=[0])
    datos.to_excel("./src/tiempos.xlsx", index=False)

    for metodo,tiempo in tiempos.items():
        print(f"Tiempo de ejecución para {metodo}: {tiempo/60:.2f} min")