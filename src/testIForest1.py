import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
import Config2
from pathlib import Path


def create_nan_csv(data_frame, name):
    """
    Crea un archivo CSV con la cantidad de valores NaN por columna de un DataFrame.

    Args:
        data_frame: DataFrame de pandas.
        name: Nombre base del archivo CSV.
    """
    nan_df = data_frame.isnull().sum()
    nan_df = nan_df[nan_df > 0]
    total = len(data_frame)
    nan_df = nan_df / total * 100
    nan_df = nan_df.to_frame().T
    base_output_path = Path('data/nan_detection')
    base_output_path.mkdir(parents=True, exist_ok=True)
    nan_df.to_csv(f'{base_output_path}/{name}_nan.csv')

def load_data(path):
    for i,label in enumerate(os.listdir(path)):
        print("i: " + str(i)  + "   /  label: " + str(label))
        print("AQ: " + str(os.path.join(path,label)) )
        df1 = pd.read_csv(os.path.join(path,label),  sep=';', decimal=',')

    

    create_nan_csv(df1, "nan")
    print(df1.head())


if __name__ == "__main__":
    x = load_data(Config2.path)
