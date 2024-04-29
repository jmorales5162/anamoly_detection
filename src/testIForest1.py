import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
import Config2
from pathlib import Path




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


def isolationForest(data):
    resultados = np.zeros((3, data.size//2))
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
               c="skyblue", marker="s", s=500)
        ax.scatter(data[:, 0], 
               data[:, 1], 
               c=range(data.size//2), marker="x",
               s=500, alpha=0.6)
        ax.set_title("Contaminación: %0.2f" % c[i], size=18, color="purple")
        ax.set_ylabel("Precio ($)", size=10)
        ax.set_xlabel("Kms recorridos", size=14)

    plt.show()



if __name__ == "__main__":
    x1, x2 = load_data(Config2.path)
    print(x1.to_numpy())
    #print(x2)
    carros = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "carros.csv" ), delimiter=",")
    print(carros)
    isolationForest(x1.to_numpy())

