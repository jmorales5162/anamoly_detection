from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from sklearn.model_selection import cross_validate
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Config 
import pandas as pd
from pathlib import Path
import joblib

class AnomaliesMethod:
    def __init__(self,df):
        self.df = df

    def isolationForest(self):
        X_scaled = StandardScaler().fit_transform(self.df)
        model = IsolationForest(contamination=0.02)
        model.fit(X_scaled)
        outliers = model.predict(X_scaled)
        self.graficarAnomalias(self.df, outliers)    

    def graficarAnomalias(self,data,outliers):
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
        X_scaled = StandardScaler().fit_transform(self.df)
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
        plt.xlabel('IRRADIATION'); plt.ylabel('DC_POWER'); plt.title('IRRADIATION vs Watts'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(X_scaled[:, 1], X_scaled[:, 2], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
        plt.scatter(anomalies[:, 1], anomalies[:, 2], color='red', marker='.', s=50, label='Anomalies')
        plt.xlabel('MODULE_TEMPERATURE'); plt.ylabel('DC_POWER'); plt.title('MODULE_TEMPERATURE vs DC_POWER'); plt.legend()
        plt.tight_layout()
        plt.show()

    def autoEncoder(self):
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
        plt.ylabel('Error de reconstrucción');
        plt.legend(["Entrenamiento", "Test", "Umbral"], loc="upper left")
        plt.show()


class RegressionMethod:
    def __init__(self , df , indepVars , depVars):
        self.df = df 
        self.indepVars = indepVars
        self.depVars = depVars

    def adestrarMetodo(self,model,nome):
        X = self.df[self.indepVars]; Y = self.df[self.depVars]
        scaler = StandardScaler()
        self.cross_validation_regression(model, scaler.fit_transform(X), scaler.fit_transform(Y), folds=10, name="Solar production", model_name=nome)
        model.fit(X, Y)
        self.graficarResultados(model, nome, X, Y)
        output_path = Path(f"models/{nome}/", f"{nome}.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        
    def cross_validation_regression(self,model, X, y, folds=5, name="", model_name=""):
        
        def smape(y_true, y_pred):
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
        
        def write_cv_results_to_file(cv_results, model_name, name):
            output_path = Path(f"graphs/models/{model_name}/{name}/cv_results.txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for metric_name, metric_values in cv_results.items():
                    if metric_name.startswith('test_'):
                        formatted_name = metric_name[5:].replace('_', ' ').capitalize()
                        mean_value = np.mean(metric_values)
                        std_dev = np.std(metric_values)
                        if "mape" in metric_name.lower() or "smape" in metric_name.lower():
                            f.write(f"{formatted_name}: {mean_value:.2f}% (+-{std_dev:.2f}%)\n")
                        else:
                            f.write(f"{formatted_name}: {mean_value:.4f} (+-{std_dev:.4f})\n")

        smape_scorer = make_scorer(smape, greater_is_better=False)  # Make smape compatible

        scoring = {
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "R2": "r2",
        }
        cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring)
        cv_results['test_MSE'] = abs(cv_results['test_MSE'])
        print("Resultados: " + str(cv_results))
        write_cv_results_to_file(cv_results, model_name, name)

    def graficarResultados(self,modelo, nombre, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        modelo.fit(X_train, Y_train)
        Y_predicted = modelo.predict(X_test)

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
        output_path = Path(f"graphs/models/{nombre}/graficaResultados.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()       
    

def process_df():
    weather_data = pd.read_csv(Config.weather_path)
    generation_data = pd.read_csv(Config.generation_path)
    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    return df_solar.drop(columns = ['DATE_TIME', 'SOURCE_KEY', 'DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'AC_POWER'])


df = process_df()
prueba = RegressionMethod(df , Config.indepVars , Config.depVars)
prueba.adestrarMetodo(Config.mr['lr_pipe'], Config.mr['lr'])


