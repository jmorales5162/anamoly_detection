from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

def graficarResultados(modelo, nombre, X, Y):
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
    output_path = Path(f"graphs/models/{model_name}/{name}/graficaResultados.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()