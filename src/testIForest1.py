import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
import Config2
from pathlib import Path
from sklearn.cluster import KMeans





def load_data(path):
    df1 = pd.read_csv(path, sep=',', decimal='.')
    selected_columns = ['W', 'radiation']
    df = df1[selected_columns] 
    df = normalize_data(df)
    return df,df1[selected_columns]



def normalize_data(df):
    new_min = 0
    new_max = 1

    # Normalize the DataFrame from -1 to 1 to 0 to 1
    normalized_df = (df + 1) / 2 * (new_max - new_min) + new_min
    return normalized_df


def d3plot():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.random.random(size=(3, 3, 3))
    z, x, y = data.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()


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
    percentile_threshold = 95
    threshold_distance = np.percentile(distances, percentile_threshold)

    # Identify anomalies
    anomalies = [data[i] for i, distance in enumerate(distances) if distance > threshold_distance]
    anomalies = np.asarray(anomalies, dtype=np.float32)

    # Printing the clusters
    colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
    plt.scatter(data[:, 0], data[:, 1], marker='.', s=50, lw=0, alpha=0.7,c=colors, edgecolor='k')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='purple', marker='.', s=50, label='Anomalies')
    plt.show()


if __name__ == "__main__":

    x1, x2 = load_data(Config2.path)
 
    #isolationForest(x1.to_numpy())
    kmeans(x1.to_numpy())


