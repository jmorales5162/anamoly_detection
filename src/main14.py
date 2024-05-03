import pandas as pd
import Config
from sklearn.pipeline import make_pipeline
from pathlib import Path
from cross_validation import cross_validation_regression
from graficar import graficarResultados
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


"""
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
"""






def randomForest(datafile, depVars, indepVars):
    df = pd.read_csv(datafile)

    X = df[indepVars]; Y = df[depVars]
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    cross_validation_regression(model, X, Y, folds=10, name="Solar production", model_name="RdmForest")
    model.fit(X, Y)

    graficarResultados(model, "RdmForest", X, Y)
    #predictions_vs_actuals(model, MODEL_NAME, name, X, y)

    #plot_residuals_vs_fitted(model, MODEL_NAME, name)

    output_path = Path("models/random_forest/", "random_forest_model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)



if __name__ == "__main__":

    # 1: Estudo do conxunto de datos

    #graficarRelacionVariables()


    # 2: Tecnicas de deteccion de anomalias

    #isolationForest(x1.to_numpy())
    #kmeans(x1.to_numpy())
    #autoEncoder2(x1.to_numpy())
    

    # 3: Modelos de regresion

    #linearRegressionModel(x1)
    #polynomialRegressionModel2()
    #gradientBoosting(Config.path, Config.depVars, Config.indepVars)
    randomForest(Config.path, Config.depVars, Config.indepVars)