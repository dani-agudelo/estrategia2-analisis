import time
from modelos.matriz import MatrizTPM
from icecream import ic
from modelos.MetricasDistancia import MetricasDistancia
import numpy as np
from itertools import chain


class AlgoritmoPrincipal:
    def __init__(self, ruta):
        self.__matriz = MatrizTPM(ruta)
        self.__emd = MetricasDistancia()
        self.__particiones_candidatas = []

    def estrategia2(self):
        self.__matriz.condiciones_de_background()
        self.__matriz.obtener_estado_nodo()
        self.__matriz.matriz_subsistema()
        self.__matriz.get_matriz_subsistema()
        self.__matriz.prueba_marginalizar_aristas()
        # t_inicio = time.time()
        # self.encontrar_particion_menor()
        # ic(self.comparar_particiones())
        # t_fin = time.time()
        # t_proceso = t_fin - t_inicio
        # ic(t_proceso)

    def encontrar_particion_menor(self):
        conjuntoA= self.__matriz.crear_conjunto_a()
        ic(conjuntoA) # (0,0), (0,1), (1,0), (1,1)
        self.algoritmo_principal(conjuntoA)

    def algoritmo_principal(self, A): 
        if(len(A) == 1):
            return
        W = [A[0]] #! debe ser aleatorio aA
        for i in range(len(A) - 1):
            mejor_iteracion = () #aB, bA, bB
            for j in list(set(A) - set(W)):
                subsistema = list(chain.from_iterable((i,) if isinstance(i[0], int) else i for i in W)) # devuelve una lista de tuplas
                u = []
                subsistema.extend(j if isinstance(j[0], tuple) else [j])
                u.extend(j if isinstance(j[0], tuple) else [j])

                resultadoEMD = self.realizar_emd(subsistema)
                resultadoEMD_nu = self.realizar_emd(u)

                resultado = resultadoEMD[0] - resultadoEMD_nu[0]

                if mejor_iteracion == () or resultado < mejor_iteracion[0]:
                    mejor_iteracion = (resultado, j)

            W.append(mejor_iteracion[1])
        
        # Tomar los dos últimos elementos de W como el par candidato
        if len(W) >= 2:
            self.__particiones_candidatas.append([resultadoEMD_nu[0], resultadoEMD_nu[1], (W[-1], W[:-1])])
            par_candidato = (W[-2], W[-1])
            # Quitar al arreglo v todos los elementos del par candidato
            A = list(set(A) - set(par_candidato))
            par_candidato_final = self.combinar_tuplas(par_candidato[0], par_candidato[1])
            A.append(par_candidato_final)

        self.algoritmo_principal(A)
    
    def realizar_emd(self, lista):
        #aA, aB (0,0), (0,1)
        matriz_normal, matriz_complemento = self.__matriz.marginalizar_normal_complemento(lista)
        est_n, est_c = self.__matriz.get_estado_inicial_n_c()
        self.__matriz.limpiar_estados_inicialies()
        resultado_tensorial = self.__matriz.producto_tensorial_matrices(matriz_normal[0], matriz_complemento[0], matriz_normal[1], matriz_complemento[1], est_n, est_c)
        resultados_lista = np.array(resultado_tensorial.iloc[0].values.tolist(), dtype='float64')
        return (self.__emd.emd_pyphi(resultados_lista, self.__matriz.get_matriz_subsistema()), resultados_lista)

    def combinar_tuplas(self, t1, t2):
        # Verificar si t1 y t2 son tuplas de tuplas (tupla con otros elementos tipo tuple)
        es_tupla_de_tuplas_1 = all(isinstance(elem, tuple) for elem in t1)
        es_tupla_de_tuplas_2 = all(isinstance(elem, tuple) for elem in t2)

        # Agrupar `t1` y `t2` en una sola tupla de tuplas según su estructura
        if not es_tupla_de_tuplas_1:
            t1 = (t1,)  # Agrupa t1 en una sola tupla si no es una tupla de tuplas
        if not es_tupla_de_tuplas_2:
            t2 = (t2,)  # Agrupa t2 en una sola tupla si no es una tupla de tuplas

        # Combinar ambos resultados en una sola tupla
        return t1 + t2

    def comparar_particiones(self):
        particion_optima = []
        menor = self.__particiones_candidatas[0][0]

        # Buscar el menor EMD
        for i in self.__particiones_candidatas:
            if i[0] < menor:
                menor = i[0]

        # Agregar las particiones con el menor EMD a la lista de particiones óptimas
        for i in self.__particiones_candidatas:
            if i[0] == menor:
                particion_optima.append(i)

        with open('archivos/particion_optima.txt', 'w') as f:
            for i in particion_optima:
                arreglo = i[1]
                np.savetxt(f, arreglo, delimiter=',', fmt='%.5f')
                f.write('\n')  # Agregar una línea en blanco entre arreglos 
        
        return particion_optima
    
    def guardar_en_archivo(self, contenido, ruta):
        with open(ruta, "w") as archivo:
            archivo.write(str(contenido))  # Escribir el contenido como texto
            
    
   
 
