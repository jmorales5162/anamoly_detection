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

from sklearn.decomposition import PCA

# AutoEncoder
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autoencoder import AutoEncoder

# Ano-malias

def plot_firms (dataframe, title, color = None):
    """ Plot firms on the 2-dimensional space """
    
    # Generate a scatter plot
    fig = px.scatter(pca_df, x="principal_component_1", y="principal_component_2", title=title, color=color)
    
    # Layout
    fig.update_layout(
        font_family='Arial Black',
        title=dict(font=dict(size=20, color='red')),
        yaxis=dict(tickfont=dict(size=13, color='black'),
                   titlefont=dict(size=15, color='black')),
        xaxis=dict(tickfont=dict(size=13, color='black'),
                   titlefont=dict(size=15, color='black')),
        legend=dict(font=dict(size=10, color='black')),
        plot_bgcolor='white'
    )
      
    return(fig)# Need to import renderers to view the plots on GitHub
# Funciona con duas variables pero con 3?
def isolationForest(path):
    data = pd.read_csv(path)
    data = data[['radiation', 'temperature']].to_numpy()

    # Standardize features
    sample_scaled = StandardScaler().fit_transform(data)# Define dimensions = 2
    pca = PCA(n_components=2)# Conduct the PCA
    principal_comp = pca.fit_transform(sample_scaled)# Convert to dataframe
    pca_df = pd.DataFrame(data = principal_comp, columns = ['principal_component_1', 'principal_component_2'])
    pca_df.head()

    # Train the model
    isf = IsolationForest(contamination=0.04)
    isf.fit(pca_df)# Predictions
    predictions = isf.predict(pca_df)

    # Extract scores
    pca_df["iso_forest_scores"] = isf.decision_function(pca_df)# Extract predictions
    pca_df["iso_forest_outliers"] = predictions# Describe the dataframe
    pca_df.describe()

    # Replace "-1" with "Yes" and "1" with "No"
    pca_df['iso_forest_outliers'] = pca_df['iso_forest_outliers'].replace([-1, 1], ["Yes", "No"])# Print the first 5 firms
    pca_df.head()

    import plotly.io as pio# Plot [1] All firms
    plot_firms(pca_df, "Figure 1: All Firms").show("png")

    


""""
def isolationForest(path):
    data = pd.read_csv(path)
    data = data[['W', 'radiation', 'temperature']].to_numpy()
    
    print(data.shape)
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
 """   


def kmeans(path, n_clusters):
    data = pd.read_csv(path)
    data = data[['W', 'radiation']].to_numpy()

    #n_clusters = 3
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



def autoEncoder(path):
    data = pd.read_csv(path)
    data = data[['W', 'radiation']].to_numpy()
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

    isolationForest(Config.path)
    #kmeans(Config.path, Config.n_clusters)
    #autoEncoder(Config.path)
    

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