import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Config
from pathlib import Path
import joblib
import datetime
from graficar import graficarResultados
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from cross_validation import cross_validation_regression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA

# AutoEncoder
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Ano-malias

def graficarAnomalias(data, outliers):
    X = df[['DC_POWER','IRRADIATION', 'MODULE_TEMPERATURE']]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data['IRRADIATION'], data['DC_POWER'], c=outliers, cmap='viridis')
    plt.scatter(data['IRRADIATION'][outliers == -1], data['DC_POWER'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('IRRADIATION')
    plt.ylabel('DC_POWER')
    plt.title('IRRADIATION vs Watts')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(data['MODULE_TEMPERATURE'], data['DC_POWER'], c=outliers, cmap='viridis')
    plt.scatter(data['MODULE_TEMPERATURE'][outliers == -1], data['DC_POWER'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('MODULE_TEMPERATURE')
    plt.ylabel('DC_POWER')
    plt.title('MODULE_TEMPERATURE vs DC_POWER')
    plt.legend()

    plt.tight_layout()
    plt.show()

def isolationForest(df):
    X_scaled = StandardScaler().fit_transform(df)
    model = IsolationForest(contamination=0.01) #contamination 1%
    model.fit(X_scaled)
    outliers = model.predict(X_scaled)
    graficarAnomalias(df, outliers)
    # Nos inventamos un 1% de datos anomalos y vemos si los clasifica bien
    # 68774 FILAS ten o dataframe total, 687 FILAS ten o dataframe anomalo
    filasDf = df.shape[0]
    num_anomalias = round(filasDf * 0.01)
    f1_results = []
    for i in range(10):
        dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                    'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                    'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
        
        Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
        df_train = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
        model.fit(df_train); Y_pred = model.predict(df_train)
        Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
        f1_results.append(f1_score(Y, Y_pred))

    f1_results = np.array(f1_results)
    mean = np.mean(f1_results); std = np.std(f1_results)
    print("media F1: " + str(mean) + " std: " + str(std))



def kmeans(df):
    X_scaled = StandardScaler().fit_transform(df)
    model = KMeans(n_clusters=Config.n_clusters)
    model.fit(X_scaled)
    cluster_labels = model.predict(X_scaled)
    cluster_centers = model.cluster_centers_
    distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X_scaled, cluster_labels)]

    percentile_threshold = 99.5
    threshold_distance = np.percentile(distances, percentile_threshold)

    anomalies = [X_scaled[i] for i, distance in enumerate(distances) if distance > threshold_distance]
    anomalies = np.asarray(anomalies, dtype=np.float32)



    colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
    plt.scatter(anomalies[:, 0], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
    plt.xlabel('IRRADIATION'); plt.ylabel('DC_POWER'); plt.title('IRRADIATION vs Watts'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 1], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
    plt.scatter(anomalies[:, 1], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
    plt.xlabel('MODULE_TEMPERATURE'); plt.ylabel('DC_POWER'); plt.title('MODULE_TEMPERATURE vs DC_POWER'); plt.legend()
    plt.tight_layout()
    plt.show()

    #Y = np.array(np.where((distances > threshold_distance), 1, 0))
    #anomalies2 = np.array(anomalies2)
    
    filasDf = df.shape[0]
    #print(" Forma das anomalias " + str(anomalies2.shape) + " Primeros 10 elementos: " + str(anomalies2[-10:]))ape[0]
    num_anomalias = round(filasDf * 0.01)
    f1_results = []
    for i in range(10):
        dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                    'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                    'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
        
        Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
        df_train = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
        model.fit(df_train)
        cluster_labels = model.predict(df_train)
        cluster_centers = model.cluster_centers_
        distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(df_train, cluster_labels)]
        threshold_distance = np.percentile(distances, percentile_threshold)
        #Y_pred = model.predict(df_train)
        Y_pred = np.array(np.where((distances > threshold_distance), 1, 0))
        #Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
        f1_results.append(f1_score(Y, Y_pred))

    #print(" Forma das anomalias " + str(Y[-10:]) + " Primeros 10 elementos: " + str(Y_pred[-10:]))
    f1_results = np.array(f1_results)
    print(f1_results)
    mean = np.mean(f1_results); std = np.std(f1_results)
    print("media F1: " + str(mean) + " std: " + str(std))



def autoEncoder(df):
    x_train, x_test, y_train, y_test = train_test_split(df[['IRRADIATION', 'MODULE_TEMPERATURE']].to_numpy(), df[['DC_POWER']].to_numpy(), test_size=0.2, random_state=111)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.fit_transform(x_test)
    Y_train_scaled = scaler.fit_transform(x_train)
    Y_test_scaled = scaler.fit_transform(x_test)
    autoencoder = models.Sequential()
    autoencoder.add(layers.Dense(1, input_shape=X_train_scaled.shape[1:], activation='relu'))
    autoencoder.add(layers.Dense(2))
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    history = autoencoder.fit(X_train_scaled, Y_train_scaled, 
          epochs=30, 
          batch_size=100,
          validation_data=(X_test_scaled, Y_test_scaled),
          shuffle=True)
    
    mse_train = tf.keras.losses.mse(autoencoder.predict(X_train_scaled), X_train_scaled)
    umbral = np.max(mse_train)

    plt.figure(figsize=(12,4))
    plt.hist(mse_train, bins=50)
    plt.xlabel("Error de reconstrucción (entrenamiento)")
    plt.ylabel("Número de datos")
    plt.axvline(umbral, color='r', linestyle='--')
    plt.legend(["Umbral"], loc="upper center")
    plt.show()
    e_test = autoencoder.predict(X_test_scaled)

    mse_test = np.mean(np.power(X_test_scaled - e_test, 2), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(range(1,X_train_scaled.shape[0]+1),mse_train,'b.')
    plt.plot(range(X_train_scaled.shape[0]+1,X_train_scaled.shape[0]+X_test_scaled.shape[0]+1),mse_test,'r.')
    plt.axhline(umbral, color='r', linestyle='--')
    plt.xlabel('Índice del dato')
    plt.ylabel('Error de reconstrucción')
    plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")
    plt.show()

    # COmo faso

    filasDf = df.shape[0]
    num_anomalias = round(filasDf * 0.01)
    f1_results = []
    for i in range(10):
        dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                    'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                    'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
        
        Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
        df_train = StandardScaler().fit_transform(pd.concat([df, dfanomalo], ignore_index=True))
        model.fit(df_train); Y_pred = model.predict(df_train)
        Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
        f1_results.append(f1_score(Y, Y_pred))

    f1_results = np.array(f1_results)
    mean = np.mean(f1_results); std = np.std(f1_results)
    print("media F1: " + str(mean) + " std: " + str(std))


def adestrarMetodo(df, model, nome, depVars, indepVars):
    X = df[indepVars]; Y = df[depVars]
    scaler = StandardScaler()
    cross_validation_regression(model, scaler.fit_transform(X), scaler.fit_transform(Y), folds=10, name="Solar production", model_name=nome)
    model.fit(X, Y)
    graficarResultados(model, nome, X, Y)
    output_path = Path(f"models/{nome}/", f"{nome}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def process_df():
    weather_data = pd.read_csv(Config.weather_path)
    generation_data = pd.read_csv(Config.generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])


if __name__ == "__main__":

    # 1: Estudo do conxunto de datos

    #graficarRelacionVariables()

    df = process_df()

    # 2: Tecnicas de deteccion de anomalias

    #isolationForest(df)
    #kmeans(df)
    autoEncoder(df)
    

    # 3: Modelos de regresion
    
    """

    methods = []
    methods.append(("linealRegression", make_pipeline(StandardScaler(), LinearRegression())))

    
    methods.append(("polynomialMethod", make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )))

    methods.append(("gradientBoostingMethod", make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    )))

    methods.append(("rdmForestMethod", make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))))


    for method in methods:
        adestrarMetodo(df, method[1], method[0], Config.depVars, Config.indepVars)
    """