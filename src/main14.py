import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Config
from pathlib import Path
import joblib
from graficar import graficarResultados
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from cross_validation import cross_validation_regression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

# AutoEncoder
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autoencoder import AutoEncoder

# Ano-malias

def graficarAnomalias(data, outliers):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data['radiation'], data['W'], c=outliers, cmap='viridis')
    plt.scatter(data['radiation'][outliers == -1], data['W'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('Radiation')
    plt.ylabel('Watts')
    plt.title('Radiation vs Watts')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(data['temperature'], data['W'], c=outliers, cmap='viridis')
    plt.scatter(data['temperature'][outliers == -1], data['W'][outliers == -1], 
            edgecolors='r', facecolors='none', s=100, label='Outliers')
    plt.xlabel('Temperature')
    plt.ylabel('Watts')
    plt.title('Temperature vs Watts')
    plt.legend()

    plt.tight_layout()
    plt.show()


def isolationForest(path):
    data = pd.read_csv(path)
    X = data[['radiation', 'temperature', 'W']]
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=0.02)
    model.fit(X_scaled)
    outliers = model.predict(X_scaled)
    graficarAnomalias(data, outliers)

    

def kmeans(path, n_clusters):
    data = pd.read_csv(path)
    X = data[['radiation', 'temperature', 'W']]
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=n_clusters)
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
    plt.xlabel('Radiation'); plt.ylabel('Watts'); plt.title('Radiation vs Watts'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 1], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
    plt.scatter(anomalies[:, 1], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
    plt.xlabel('Temperature'); plt.ylabel('Watts'); plt.title('Temperature vs Watts'); plt.legend()
    plt.tight_layout()
    plt.show()




def autoEncoder(path):

    data = pd.read_csv(path)
    data = data[['W', 'radiation', 'temperature']]
    x_train, x_test, y_train, y_test = train_test_split(data[['radiation', 'temperature']].to_numpy(), data[['W']].to_numpy(), test_size=0.2, random_state=111)
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
    plt.ylabel('Error de reconstrucción');
    plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")
    plt.show()



def adestrarMetodo(datafile, metodo, nome, depVars, indepVars):
    df = pd.read_csv(datafile); X = df[indepVars]; Y = df[depVars]
    model = make_pipeline(StandardScaler(), metodo)
    cross_validation_regression(model, X, Y, folds=10, name="Solar production", model_name=nome)
    model.fit(X, Y)
    graficarResultados(model, nome, X, Y)
    output_path = Path(f"models/{nome}/", f"{nome}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def graficarRelacionVariables(path):
    df = pd.read_csv(path)
    print("Max: " + str(df[['radiation']].max()))
    X_radiation = df[['radiation']].to_numpy()
    X_temperature = df[['temperature']].to_numpy()
    Y_potencia = df[['W']].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(X_radiation, Y_potencia.flatten(), label="Prediction", color='red')
    ax1.set_xlabel('Radiacion'); ax1.set_ylabel('Produccion')
    ax1.set_title('Produccion Solar')

    ax2.scatter(X_temperature, Y_potencia.flatten(), label="Real", color='blue')
    ax2.set_xlabel('Temperatura'); ax2.set_ylabel('Produccion')
    ax2.set_title('Produccion Solar')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 1: Estudo do conxunto de datos

    graficarRelacionVariables(Config.path)


    # 2: Tecnicas de deteccion de anomalias

    #isolationForest(Config.path)
    #kmeans(Config.path, Config.n_clusters)
    #autoEncoder(Config.path)
    

    # 3: Modelos de regresion




    methods = []
    methods.append(("linealRegression", make_pipeline(StandardScaler(), LinearRegression())))

    """
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
    """
    


    #for method in methods:
    #    adestrarMetodo(Config.path, method[1], method[0], Config.depVars, Config.indepVars)
       

