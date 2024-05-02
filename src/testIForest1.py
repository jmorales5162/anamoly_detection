import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import pandas as pd
import os
import Config2
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib


# AutoEncoder
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autoencoder import AutoEncoder





def load_data(path):
    df1 = pd.read_csv(path, sep=',', decimal='.')
    selected_columns = ['W', 'radiation']
    df = df1[selected_columns] 
    df = normalize_data(df)
    return df


def normalize_data(df):
    new_min = 0
    new_max = 1

    # Normalize the DataFrame from -1 to 1 to 0 to 1
    #normalized_df = (df + 1) / 2 * (new_max - new_min) + new_min
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

"""
def d3plot():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.random.random(size=(3, 3, 3))
    z, x, y = data.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
"""

def isolationForest(data):
    resultados = np.zeros((3, data.shape[0])) # (3,52547)
    c = [0.01, 0.05, 0.1] 
    for i in range(len(c)):
        modelo = IsolationForest(contamination=c[i]).fit(data)
        resultados[i] = modelo.predict(data)
    
    # Graficar datos anomalos 
    plt.set_cmap("jet")
    fig = plt.figure(figsize=(13, 4))

    for i in range(len(c)):    
        ax = fig.add_subplot(1, 3, i+1)
        ax.scatter(data[resultados[i]==-1][:, 0], 
               data[resultados[i]==-1][:, 1], 
               c="skyblue", marker="s", s=50)
        ax.scatter(data[:, 0], 
               data[:, 1], 
               c=range(data.size//2), marker="x",
               s=50, alpha=0.6)
        ax.set_title("Contaminación: %0.2f" % c[i], size=18, color="purple")
        ax.set_ylabel("Radiacion", size=8)
        ax.set_xlabel("Watts", size=8)

    plt.show()




def kmeans(data):
    n_clusters = 3
    n_clusters_to_detect = n_clusters # Number of clusters (including anomalies)
    kmeans = KMeans(n_clusters=n_clusters_to_detect)
    kmeans.fit(data)
    # Predict cluster labels
    cluster_labels = kmeans.predict(data)

    # Find the cluster centers
    cluster_centers = kmeans.cluster_centers_
    # Calculate the distance from each point to its assigned cluster center
    distances = [np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(data, cluster_labels)]
    # Define a threshold for anomaly detection (e.g., based on the distance percentile)
    percentile_threshold = 99.5
    threshold_distance = np.percentile(distances, percentile_threshold)

    # Identify anomalies
    anomalies = [data[i] for i, distance in enumerate(distances) if distance > threshold_distance]
    anomalies = np.asarray(anomalies, dtype=np.float32)

    # Printing the clusters
    colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
    plt.scatter(data[:, 0], data[:, 1], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', marker='.', s=50, label='Anomalies')
    plt.show()



def autoEncoder(data):
    x_train, x_test, y_train, y_test = train_test_split(data.values, data.values[:,0:1], test_size=0.2, random_state=111)
    model = AutoEncoder()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
    model.compile(optimizer='adam', loss="mae")
    history = model.fit(normal_train_data, normal_train_data, epochs=50, batch_size=120,
                    validation_data=(train_data_scaled[:,1:], train_data_scaled[:, 1:]),
                    shuffle=True,
                    callbacks=[early_stopping]
                    )
    encoder_out = model.encoder(normal_test_data).numpy() #8 unit representation of data
    decoder_out = model.decoder(encoder_out).numpy()
    plt.plot(normal_test_data[0], 'b')
    plt.plot(decoder_out[0], 'r')
    plt.title("Model performance on Normal data")
    plt.show()

    encoder_out_a = model.encoder(anomaly_test_data).numpy() #8 unit representation of data
    decoder_out_a = model.decoder(encoder_out_a).numpy()
    plt.plot(anomaly_test_data[0], 'b')
    plt.plot(decoder_out_a[0], 'r')
    plt.title("Model performance on Anomaly Data")
    plt.show()

    reconstruction = model.predict(normal_test_data)
    train_loss = tf.keras.losses.mae(reconstruction, normal_test_data)
    plt.hist(train_loss, bins=50)

    threshold = np.mean(train_loss) + 2*np.std(train_loss)
    reconstruction_a = model.predict(anomaly_test_data)
    train_loss_a = tf.keras.losses.mae(reconstruction_a, anomaly_test_data)
    plt.hist(train_loss_a, bins=50)
    plt.title("loss on anomaly test data")
    plt.show()

    plt.hist(train_loss, bins=50, label='normal')
    plt.hist(train_loss_a, bins=50, label='anomaly')
    plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
    plt.legend(loc='upper right')
    plt.title("Normal and Anomaly Loss")
    plt.show()


def autoEncoder2(data):
    x_train, x_test, y_train, y_test = train_test_split(data, data[:,0:1], test_size=0.2, random_state=111)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.fit_transform(x_test)
    autoencoder = models.Sequential()
    autoencoder.add(layers.Dense(1, input_shape=X_train_scaled.shape[1:], activation='relu'))
    autoencoder.add(layers.Dense(2))
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    history = autoencoder.fit(X_train_scaled, X_train_scaled, 
          epochs=30, 
          batch_size=100,
          validation_data=(X_test_scaled, X_test_scaled),
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



def linearRegressionModel(data):
    reg = LinearRegression()
    X = np.array([data['radiation'].values])
    Y = np.array([data['W'].values])
    X = X.transpose(); Y = Y.transpose()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print("Shapes: X=" + str(X_train.shape) + " Y=" + str(y_train.shape))
    reg = reg.fit(X_train,y_train)
    Y_pred = reg.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(X_test, Y_pred.flatten(), label="Prediction", color='red')
    ax1.set_xlabel('Radiacion'); ax1.set_ylabel('Produccion')
    ax1.set_title('Produccion Solar')

    ax2.scatter(X_test, y_test.flatten(), label="Real", color='blue')
    ax2.set_xlabel('Radiacion'); ax2.set_ylabel('Produccion')
    ax2.set_title('Produccion Solar')
    plt.tight_layout()
    plt.show()



def polynomialRegressionModel2():
    df = normalize_data(pd.read_csv(Config2.path))
    Y = df[['W']]
    X = df[['radiation', 'temperature']]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    poly_features = poly.fit_transform(X_train)
    
    #X_train, X_test, Y_train, Y_test = train_test_split(poly_features, Y, test_size=0.3, random_state=42)
    print("Shapes X: " + str(X_test.shape) + "Shapes Y: " + str(Y_test.shape))

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, Y_train)
    Y_predicted = poly_reg_model.predict(X_test)
    poly_reg_rmse = np.sqrt(mean_squared_error(Y_test, Y_predicted))
    print("Resultados: " + str(poly_reg_rmse))



    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
    #print("Shapes X: " + str(X_test.to_numpy()[:,0:1].shape) + "Shapes Y: " + str(Y_predicted.shape))
    #Sprint(X_test[4,:])
    X_test_rad = X_test.to_numpy()[:,0:1]
    X_test_temp = X_test.to_numpy()[:,1:2]

    ax1.scatter(X_test_rad, Y_predicted, label="Prediccion", color='red')
    ax1.set_xlabel('Radiacion'); ax1.set_ylabel('Prediccion Watts')

    ax2.scatter(X_test_rad, Y_test, label="Real", color='orange')
    ax2.set_xlabel('Radiacion'); ax2.set_ylabel('Produccion Watts Real')

    ax3.scatter(X_test_temp, Y_predicted, label="Prediccion", color='blue')
    ax3.set_xlabel('Temperatura'); ax3.set_ylabel('Prediccion Watts')
    #ax3.set_title('Produccion Solar')

    ax4.scatter(X_test_temp, Y_test, label="Prediccion", color='cyan')
    ax4.set_xlabel('Temperatura'); ax4.set_ylabel('Produccion Watts Real')
    #ax4.set_title('Produccion Solar')
    plt.tight_layout()
    plt.show()



def graficarRelacionVariables():
    df = normalize_data(pd.read_csv(Config2.path))
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


def gradientBoosting():
    df = normalize_data(pd.read_csv(Config2.path))
    Y = df[['W']]
    X = df[['radiation', 'temperature']]
    print(X.head())
    print(Y.head())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    gbc=GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,random_state=100,max_features=5 )
    gbc.fit(X_train, Y_train)
    Y_predicted = gbc.predict(X_test)
    poly_reg_rmse = np.sqrt(mean_squared_error(Y_test, Y_predicted))
    graficar_prediccions(X_test, Y_predicted, Y_test)


def graficar_prediccions(X_test, Y_predicted, Y_test):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))

    X_test_rad = X_test.to_numpy()[:,0:1]
    X_test_temp = X_test.to_numpy()[:,1:2]

    ax1.scatter(X_test_rad, Y_predicted, label="Prediccion", color='red')
    ax1.set_xlabel('Radiacion'); ax1.set_ylabel('Prediccion Watts')

    ax2.scatter(X_test_rad, Y_test, label="Real", color='orange')
    ax2.set_xlabel('Radiacion'); ax2.set_ylabel('Produccion Watts Real')

    ax3.scatter(X_test_temp, Y_predicted, label="Prediccion", color='blue')
    ax3.set_xlabel('Temperatura'); ax3.set_ylabel('Prediccion Watts')

    ax4.scatter(X_test_temp, Y_test, label="Prediccion", color='cyan')
    ax4.set_xlabel('Temperatura'); ax4.set_ylabel('Produccion Watts Real')

    plt.tight_layout()
    plt.show()


def perform_regression_cv(model, X, y, folds=5, name="", model_name=""):
    from sklearn.metrics import (
        make_scorer,
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
    )
    def smape(y_true, y_pred):
        return 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    )
    def write_cv_results_to_file(cv_results, model_name, name):
        output_path = Path(f"graphs/models/{model_name}/{name}/cv_results.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for metric_name, metric_values in cv_results.items():
                if metric_name.startswith('test_'):
                    # Formatear el nombre de la métrica para mejorar la legibilidad
                    formatted_name = metric_name[5:].replace('_', ' ').capitalize()
                    # Calcular promedio y desviación estándar
                    mean_value = np.mean(metric_values)
                    std_dev = np.std(metric_values)
                    # Ajustar el formato del mensaje según el tipo de métrica
                    if "mape" in metric_name.lower() or "smape" in metric_name.lower():
                        f.write(f"{formatted_name}: {mean_value:.2f}% (+-{std_dev:.2f}%)\n")
                    else:
                        f.write(f"{formatted_name}: {mean_value:.4f} (+-{std_dev:.4f})\n")

    smape_scorer = make_scorer(smape, greater_is_better=False)  # Make smape compatible

    scoring = {
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False,
        ),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "SMAPE": smape_scorer,
        "R2": "r2",
    }
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring)

    write_cv_results_to_file(cv_results, model_name, name)


def predictions_vs_actuals(model, model_name, name, X, y):
    # Predictions vs Actuals
    predictions = model.predict(X)
    plt.scatter(y, predictions)
    plt.xlabel("Actual values")
    plt.ylabel("Predictions")
    plt.title(f"Predictions vs. Actuals for {name}")
    output_path = Path(f"graphs/models/{model_name}/{name}/predictions_vs_actuals.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def randomForest(datafile, dep_vars, indep_vars):
    df = pd.read_csv(datafile)

    X = df[indep_vars]  # Independent variables
    y = df[dep_vars]  # Dependent variable

    # Perform the linear regression
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))

    name = "RdmForest"
    MODEL_NAME = "RdmForest"
    perform_regression_cv(model, X, y, folds=10, name=name, model_name=MODEL_NAME)

    model.fit(X, y)

    predictions_vs_actuals(model, MODEL_NAME, name, X, y)

    #plot_residuals_vs_fitted(model, MODEL_NAME, name)

    output_path = Path("models/random_forest/", 'random_forest_model.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)



def load_data2(path):
    df1 = pd.read_csv(path)
    return normalize_data(df1[['W', 'radiation', 'temperature']])




if __name__ == "__main__":

    x1 = load_data(Config2.path)

    # Estudio do conxunto de datos
    #graficarRelacionVariables()


    # Tecnicas de deteccion de anomalias

    #isolationForest(x1.to_numpy())
    #kmeans(x1.to_numpy())
    #autoEncoder2(x1.to_numpy())
    


    # Regresion

    #linearRegressionModel(x1)
    #polynomialRegressionModel2()
    gradientBoosting(Config2.path, "W", ["radiation", "temperature"])
    randomForest(Config2.path, "W", ["radiation", "temperature"])