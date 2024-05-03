import pandas as pd
import Config
from pathlib import Path
import joblib
from graficar import graficarResultados
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from cross_validation import cross_validation_regression


def randomForest(datafile, depVars, indepVars):
    df = pd.read_csv(datafile)

    X = df[indepVars]; Y = df[depVars]
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    cross_validation_regression(model, X, Y, folds=10, name="Solar production", model_name="RdmForest")
    model.fit(X, Y)

    graficarResultados(model, "RdmForest", X, Y)

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