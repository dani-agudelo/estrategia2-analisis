import pandas as pd
from modelos.matriz import MatrizTPM
import numpy as np

class LectorExcel:
    def __init__(self, ruta = 'archivos\estado_nodo_15.xlsx'):
        self.__ruta = ruta
        self.__matrices = []

    def leer(self):
        df = pd.read_excel(self.__ruta, header=None)
        arr = df.values
        for col in range(arr.shape[1]):
            # Seleccionamos la columna
            column = arr[:, col]
            # Creamos una matriz con la misma cantidad de filas que la columnas y dos columnas.
            new_mat = np.zeros((arr.shape[0], 2))
            # Llenamos la primera columna con el complemento de la columna original
            new_mat[:, 0] = 1 - column
            # Llenamos la segunda columna con la columna original
            new_mat[:, 1] = column
            # Agregamos la nueva matriz a la lista de tensores
            self.__matrices.append(new_mat)
        print(len(self.__matrices))
        return self.__matrices
