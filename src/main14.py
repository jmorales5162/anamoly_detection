import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Config
from pathlib import Path
import joblib
from graficar import graficarResultados
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from cross_validation import cross_validation_regression

# Ano-malias
def normalize_data(df):
    new_min = 0; new_max = 1
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


def isolationForest(path):
    data = pd.read_csv(path).to_numpy()
    resultados = np.zeros((3, data.shape[0])) # (3,52547)
    c = [0.01, 0.05, 0.1] 
    for i in range(len(c)):
        modelo = IsolationForest(contamination=c[i]).fit(data)
        resultados[i] = modelo.predict(data)
    
    print(resultados.shape)
    print(data.shape)
    
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
    


def kmeans(path):
    data = pd.read_csv(path)
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



def adestrarMetodo(datafile, metodo, nome, depVars, indepVars):
    df = pd.read_csv(datafile); X = df[indepVars]; Y = df[depVars]
    model = make_pipeline(StandardScaler(), metodo)
    cross_validation_regression(model, X, Y, folds=10, name="Solar production", model_name=nome)
    model.fit(X, Y)
    graficarResultados(model, nome, X, Y)
    output_path = Path(f"models/{nome}/", f"{nome}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":

    # 1: Estudo do conxunto de datos

    #graficarRelacionVariables()


    # 2: Tecnicas de deteccion de anomalias
    #df1 = pd.read_csv(path)
    #normalize_data(df1[['W', 'radiation', 'temperature']])
    isolationForest(Config.path)
    #kmeans(x1.to_numpy())
    #autoEncoder2(x1.to_numpy())
    

    # 3: Modelos de regresion

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

    
    #for method in methods:
    #    adestrarMetodo(Config.path, method[1], method[0], Config.depVars, Config.indepVars)