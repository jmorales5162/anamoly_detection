

"""
AnomaliesMethods
================
"""

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import f1_score

class AnomaliesMethod:
    """Representa las técnicas de detección de anomalías empleadas.

    :param df: Dataframe de datos a analizar
    :type df: dataframe

    """
    def __init__(self,df):
        """Inicializa un objeto de la clase AnomaliesMethod."""
        self.df = df

    def isolationForest(self , contamination):
        """
        - Normaliza los datos de entrada
        - Entrena y predice un modelo IsolationForest
        - Grafica las anomalias encontradas en el conjunto de datos 

        :param contamination: Grado de contaminación del modelo [0-1].
        :type contamination: float
        """
        X_scaled = StandardScaler().fit_transform(self.df)
        model = IsolationForest(contamination=contamination)
        model.fit(X_scaled)
        outliers = model.predict(X_scaled)
        #self.graficarAnomalias(self.df, outliers)

        # Nos inventamos un 1% de datos anomalos y vemos si los clasifica bien
        # 68774 FILAS ten o dataframe total, 687 FILAS ten o dataframe anomalo
        filasDf = self.df.shape[0]
        num_anomalias = round(filasDf * 0.01)
        f1_results = []
        for i in range(10):
            dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                        'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                        'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
            
            Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
            df_train = StandardScaler().fit_transform(pd.concat([self.df, dfanomalo], ignore_index=True))
            model.fit(df_train); Y_pred = model.predict(df_train)
            Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
            f1_results.append(f1_score(Y, Y_pred))

        f1_results = np.array(f1_results)
        mean = np.mean(f1_results); std = np.std(f1_results)
        print("media F1: " + str(mean) + " std: " + str(std))   

    def graficarAnomalias(self,data,outliers):
        """
        - Grafica las anomalias encontradas en el conjunto de datos 

        :param data: Datos a graficar.
        :param outliers: Anomalías detectadas.
        :type data: float 
        :type outliers: float
        """
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

    def kmeans(self, n_clusters):
        """
        - Normaliza los datos de entrada
        - Entrena y predice un modelo KMeans
        - Grafica las anomalias encontradas en el conjunto de datos 

        :param n_clusters: Cantidad de clusters en los que se desea dividir los datos.
        :type n_clusters: int
        """
        X_scaled = StandardScaler().fit_transform(self.df)
        model = KMeans(n_clusters=n_clusters)
        model.fit(X_scaled)
        cluster_labels = model.predict(X_scaled)
        cluster_centers = model.cluster_centers_
        distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X_scaled, cluster_labels)]

        percentile_threshold = 99.0
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

        filasDf = self.df.shape[0]
        #print(" Forma das anomalias " + str(anomalies2.shape) + " Primeros 10 elementos: " + str(anomalies2[-10:]))ape[0]
        num_anomalias = round(filasDf * 0.01)
        f1_results = []
        for i in range(10):
            dfanomalo = pd.DataFrame({'DC_POWER':np.random.randint(0.0,28,size=num_anomalias),
                        'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                        'IRRADIATION':np.random.randint(0.0,171,size=num_anomalias)})
            
            Y = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
            df_train = StandardScaler().fit_transform(pd.concat([self.df, dfanomalo], ignore_index=True))
            model.fit(df_train)
            cluster_labels = model.predict(df_train)
            cluster_centers = model.cluster_centers_
            distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(df_train, cluster_labels)]
            threshold_distance = np.percentile(distances, percentile_threshold)
            #Y_pred = model.predict(df_train)
            Y_pred = np.array(np.where((distances > threshold_distance), 1, 0))
            #Y_pred[Y_pred == 1] = 0; Y_pred[Y_pred == -1] = 1
            f1_results.append(f1_score(Y, Y_pred))

        f1_results = np.array(f1_results)
        print(f1_results)
        mean = np.mean(f1_results); std = np.std(f1_results)
        print("media F1: " + str(mean) + " std: " + str(std))

    def autoEncoder(self):
        """
        - Normaliza los datos de entrada
        - Entrena y predice una red autoncoder
        - Grafica las anomalias encontradas en el conjunto de datos 

        """
        x_train, x_test, y_train, y_test = train_test_split(self.df[['IRRADIATION', 'MODULE_TEMPERATURE']].to_numpy(), self.df[['DC_POWER']].to_numpy(), test_size=0.2, random_state=111)
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

        filasDf = self.df.shape[0]
        num_anomalias = round(filasDf * 0.01)
        f1_results = []
        for i in range(10):
            dfanomalo = pd.DataFrame({'IRRADIATION':np.random.randint(30,100,size=num_anomalias),
                        'MODULE_TEMPERATURE':np.random.randint(0.0,56,size=num_anomalias),
                        'DC_POWER':np.random.randint(0,10,size=num_anomalias)})
            salidas = StandardScaler().fit_transform(pd.concat([self.df, dfanomalo], ignore_index=True)[['DC_POWER']])
            X_test_a = StandardScaler().fit_transform(pd.concat([self.df, dfanomalo], ignore_index=True).drop(columns=['DC_POWER']))
            Y_test_a = np.concatenate([np.zeros(filasDf, dtype=int), np.ones(num_anomalias, dtype=int)])
            e_test_a = autoencoder.predict(X_test_a)
            mse_test_a = np.mean(np.power(salidas - e_test_a, 2), axis=1)
            Y_pred_a = np.array(np.where((mse_test_a > umbral), 1, 0))
            f1_results.append(f1_score(Y_test_a, Y_pred_a))

        f1_results = np.array(f1_results)
        print(f1_results)
        mean = np.mean(f1_results); std = np.std(f1_results)
        print("media F1: " + str(mean) + " std: " + str(std))
