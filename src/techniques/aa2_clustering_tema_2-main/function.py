import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_clusters(data, labels, centers=None, params=None):
    plt.figure(figsize=params['figsize'])
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(params['title'] , fontsize = 12)
    plt.xticks(params['xticks'], fontsize=params['fsxtick'])
    plt.yticks(params['yticks'], fontsize=params['fsytick'])
    
    if centers is not None:
        for center in centers:
            plt.scatter(center[0], center[1], c='red', marker='o', s=20, edgecolors='black')

    unique_labels = np.unique(labels)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(label)), label=label) for label in unique_labels]
    legend1 = plt.legend(handles=handles, title="Grupos", loc=params['legend_loc'])


def plot_figure(x , y , params):
    plt.figure(figsize=params['figsize'])
    #plt.plot(x, y, marker='o' , linewidth=0.5)
    plt.plot(x, y, marker=params['maker'] )
    plt.title(params['title']   , fontsize = 10)
    plt.xlabel(params['xlabel'] , fontsize = params['fsxlbl'])
    plt.ylabel(params['ylabel'] , fontsize = params['fsylbl'])

    plt.xticks(params['xticks'] , fontsize = params['fsxtick'])
    plt.yticks(params['yticks'] , fontsize = params['fsytick'])

    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_two_figure(k_values , inertia_values , silhouette_scores):
    fig, ax1 = plt.subplots()       #se crean las figuras y los ejes 

    color = 'tab:blue'               # Graficar la inercia en el eje y izquierdo
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inercia', color=color)
    ax1.plot(k_values, inertia_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    step = 1
    x_ticks = np.arange(min(k_values), max(k_values) + step, step)
    ax1.set_xticks(x_ticks)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Coeficiente de silhouette', color=color)
    ax2.plot(k_values, silhouette_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

#Definimos una fucion para calcular los valores de inercia y la puntuación de silueta
def k_medias_func(data , k_values):
    inertia_values, silhouette_scores = [[] for _ in range(2)]
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        if k == 1:
            print("Para poder aplicar la puntuación de silueta debe de haber al menos dos cluster (k).")
        elif k > 1:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    return kmeans , inertia_values , silhouette_scores  

def count_labels(labels):
    num_grupos = len(np.unique(labels)) - (1 if -1 in labels else 0)
    num_ruido = sum(1 for label in labels if label == -1)
    return num_grupos , num_ruido