import pandas as pd
import Config






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
    gradientBoosting(Config2.path, "W", ["radiation", "temperature"])
    randomForest(Config2.path, "W", ["radiation", "temperature"])