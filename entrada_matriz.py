import numpy as np
import pandas as pd
from modelos.AlgoritmoPrincipal import AlgoritmoPrincipal
from modelos.LectorExcel import LectorExcel
from icecream import ic
from modelos.matriz import MatrizTPM
from numpy.typing import NDArray
from functools import reduce

def main():
    algoritmo = AlgoritmoPrincipal('archivos/matrizGuia.csv')
    # algoritmo2 = AlgoritmoPrincipal('archivos\matriz_6_variables.csv')
    # algoritmo.estrategia1()
    excel = LectorExcel()
    tensor = excel.leer() # listado de np NDarray 
    tensor_invertido = [np.array(m) for m in reversed(tensor)]

    # Aplicar el producto tensorial en el orden inverso
    resultados = []
    for filas in zip(*tensor_invertido):  # Itera fila a fila de cada matriz en el tensor invertido
        # Multiplica los vectores en el orden inverso (de último a primero)
        producto = reduce(lambda x, y: np.kron(x, y), filas)  # `np.kron` es el producto tensorial
        resultados.append(producto)

    # Convertimos la lista `resultados` a un arreglo numpy con la forma final
    resultado_matriz = np.array(resultados)
    # exportar a csv
    pd.DataFrame(resultado_matriz).to_csv('resultado_15.csv', index=False, header=False)


# def product(
#     arrays: list[tuple[tuple[int, ...], NDArray[np.float64]]], le: bool = True
# ) -> tuple[tuple[int, ...], NDArray[np.float64]]:
#     # return reduce(lambda x, y: np.kron
#     return (
#         arrays[0]
#         if len(arrays) == 1
#         else reduce(
#             lambda x, y: bin_prod(x, y, le),
#             arrays,
#         )
#     )

def bin_prod(
    idx_dist_u: tuple[tuple[int, ...], np.ndarray],
    idx_dist_v: tuple[tuple[int, ...], np.ndarray],
    le: bool = True,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Returns the binary product of two arrays."""
    print(idx_dist_u)
    print(idx_dist_v)
    u_idx, u = idx_dist_u
    v_idx, v = idx_dist_v
    print(u)
    print(v)
    u = u.flatten()
    v = v.flatten()
    d_len = len(u_idx) + len(v_idx)
    result = np.zeros(2**d_len, dtype=np.float64)
    endian_keys = lil_endian(d_len)
    df_result = pd.DataFrame([result], columns=endian_keys)
    combined_idx = tuple(sorted(set(u_idx) | set(v_idx)))
    for key in endian_keys:
        u_key = ''.join(key[combined_idx.index(i)] for i in u_idx)
        v_key = ''.join(key[combined_idx.index(i)] for i in v_idx)
        u_val = u[int(u_key[::-1], 2)]
        v_val = v[int(v_key[::-1], 2)]
        df_result.at[0, key] = u_val * v_val
    return combined_idx, df_result.values

def lil_endian(n: int) -> list[str]:
    """Generate a list of strings representing the numbers in
    little-endian for indices in ``range(2**n)``.
    """
    return [bin(i)[2:].zfill(n)[::-1] for i in range(2**n)]

if __name__ == '__main__':
    main()