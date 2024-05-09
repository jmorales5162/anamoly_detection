"""
RegressionMethods
=================
"""

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RegressionMethod:
    """Representa las métodos de regresión empleadas.

    :param df: Dataframe de datos a analizar
    :param indepVars: Variables independientes
    :param depVars: Variables dependientes

    :type df: dataframe
    :type df: list
    :type df: list

    """
    def __init__(self , df , indepVars , depVars):
        """Inicializa un objeto de la clase AnomaliesMethod."""
        self.df = df 
        self.indepVars = indepVars
        self.depVars = depVars

    def adestrarMetodo(self,model,nome):
        """
        - Normaliza los datos de entrada
        - Realiza la validación cruzada de los datos
        - Entrena el modelo
        - Grafica los resultados  

        :param model: Modelo a procesar
        :param nome: Nombre del modelo a procesar
 
        :type model: objeto Pipeline
        :type nome: string
        """
        X = self.df[self.indepVars]; Y = self.df[self.depVars]
        scaler = StandardScaler()
        self.cross_validation_regression(model, scaler.fit_transform(X), scaler.fit_transform(Y), folds=10, name="Solar production", model_name=nome)
        model.fit(X, Y)
        self.graficarResultados(model, nome, X, Y)
        output_path = Path(f"models/{nome}/", f"{nome}.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        
    def cross_validation_regression(self,model, X, y, folds=5, name="", model_name=""):
        """
        - Realiza la validación cruzada
        - Guarda metricas en un fichero .txt

        :param model: Modelo a procesar
        :param X: Datos de entrada
        :param y: Datos de salida
        :param model_name: Nombre del modelo

        :type model: objeto Pipeline
        :type model_name: string
        """
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
        """
        - Grafica los resultados 

        :param modelo: Modelo a procesar
        :param nombre: Nombre del modelo
        :param X: Datos de entrada
        :param Y: Datos de salida

        :type modelo: objeto Pipeline
        :type nombre: string
        """
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
    

