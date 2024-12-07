import time
from modelos.matriz import MatrizTPM
from icecream import ic
from modelos.MetricasDistancia import MetricasDistancia
import numpy as np
import random
from itertools import chain


class AlgoritmoPrincipal:
    def __init__(self, ruta):
        self.__matriz = MatrizTPM(ruta)
        self.__emd = MetricasDistancia()
        self.__lista_biparticiones = []
        self.__lista_kparticiones = []
        self.__menor_biparticion = None

    def estrategia2(self):
        self.__matriz.condiciones_de_background()
        self.__matriz.obtener_estado_nodo()
        self.__matriz.matriz_subsistema()
        self.__matriz.get_matriz_subsistema()
        self.__matriz.matriz_conexiones()
        t_inicio = time.time()
        self.encontrar_particion_menor()
        t_fin = time.time()
        t_proceso = t_fin - t_inicio
        ic(t_proceso)
        self.guardar_biparticiones()
        self.guardar_kparticiones()

    def encontrar_particion_menor(self):
        conjuntoA= self.__matriz.crear_conjunto_a()
        self.algoritmo_principal(conjuntoA, 1)

    def algoritmo_principal(self, A, counter): 
        if(len(A) == 1):
            ic(self.__menor_biparticion)
            return
        posicion_aleatoria = random.randint(0, len(A) - 1)
        W = [A[posicion_aleatoria]]
        
        for i in range(len(A) - 1):
            mejor_iteracion = None #aB, bA, bB
            for j in list(set(A) - set(W)):
                subsistema = list(chain.from_iterable((i,) if isinstance(i[0], int) else i for i in W)) # devuelve una lista de tuplas
                u = []
                subsistema.extend(j if isinstance(j[0], tuple) else [j]) # [(0,1), (0,0), (1,2)]
                u.extend(j if isinstance(j[0], tuple) else [j])
                
                resultado_union = self.realizar_emd(subsistema)
                resultado_u = self.realizar_emd(u)

                diferencia = resultado_union[0] - resultado_u[0]
                
                #* verificar si hay particion
                particion = self.revisar_particion(subsistema)
                cantidad_particiones = len(particion)
                
                #* creamos una lista de diccionarios que va a guardar toda la info para comparar
                #* guardamos resultadoEMD, resultadoEMD_nu, resultado, particion, y el j. Cada diccionario es de una iteracion
                if cantidad_particiones >= 2:
                    dict_particion = {
                        'particion': particion,
                        'resultado_union': resultado_union,
                        'resta_union_u': diferencia,
                        'N_particiones': cantidad_particiones,
                        'iteracion': i,
                        'nivel_recursion': counter,
                        'dist_original': self.__matriz.get_matriz_subsistema()
                    }
                    self.__lista_biparticiones.append(dict_particion) if cantidad_particiones == 2 else self.__lista_kparticiones.append(dict_particion)

                    if cantidad_particiones == 2:
                        if self.__menor_biparticion:
                            if resultado_union[0] < self.__menor_biparticion['resultado_union'][0]:
                                self.__menor_biparticion = dict_particion
                            elif resultado_union[0] == self.__menor_biparticion['resultado_union'][0]:
                                if diferencia < self.__menor_biparticion['resta_union_u']:
                                    self.__menor_biparticion = dict_particion
                        else:
                            self.__menor_biparticion = dict_particion

                if mejor_iteracion is None or diferencia < mejor_iteracion['resta_union_u']:
                    mejor_iteracion = {
                        'resultado_union': resultado_union,
                        'resta_union_u': diferencia,
                        'arista': j
                    }
                elif diferencia == mejor_iteracion['resta_union_u']:
                    if resultado_union[0] < mejor_iteracion['resultado_union'][0]:
                        mejor_iteracion = {
                            'resultado_union': resultado_union,
                            'resta_union_u': diferencia,
                            'arista': j
                        }

            W.append(mejor_iteracion['arista'])
        
        
        if len(W) >= 2:
            par_candidato = (W[-2], W[-1]) 
            # Quitar al arreglo v todos los elementos del par candidato
            A = list(set(A) - set(par_candidato))
            par_candidato_final = self.combinar_tuplas(par_candidato[0], par_candidato[1])
            A.append(par_candidato_final)
            
        self.algoritmo_principal(A, counter + 1)
    
    '''
    Marginaliza segun la lista y devuelve el emd y la distribucion en una tupla
    la primera posicion es el emd y la segunda es la distribucion
    '''
    def realizar_emd(self, lista):
        # Obtenemos la distribución experimental
        experimental = self.__matriz.marginalizar_aristas(lista)
        experimental = np.array(experimental.iloc[0].values.tolist(), dtype='float64')
        return (self.__emd.emd_pyphi(experimental, self.__matriz.get_matriz_subsistema()), experimental)
    
    '''
    Recorre las aristas del subsistema y las elimina de la matriz de conexiones
    Retorna la cantidad de particiones, si no hay bi o k-particion, retorna 1
    '''
    def revisar_particion(self, subsistema):
        matriz_conexiones = self.__matriz.matriz_conexiones()
        #* Aplicamos las desconexiones indicadas en el subsistema
        for arista in subsistema:
            # matriz_conexiones[arista[0]][arista[1]] = 0
            matriz_conexiones.loc[arista[0], arista[1]] = 0

        dict_conexiones = {
            row_label: set(matriz_conexiones.columns[matriz_conexiones.loc[row_label] == 1])
            for row_label in matriz_conexiones.index
        }

        dict_conexiones = self.combinar_grupos(matriz_conexiones, dict_conexiones)
        
        return dict_conexiones

    '''
    Construye grupos a partir de la matriz de conexiones y los combina si tienen elementos en comun
    '''
    def combinar_grupos(self, matriz, diccionario):
        # Grupos finales: lista donde cada elemento será una tupla (filas, columnas)
        grupos = []

        # Convertir el diccionario en una lista de tuplas (fila, columnas asociadas)
        entradas = [(set([fila]), set(columnas)) for fila, columnas in diccionario.items()]
        
        while entradas:
            filas_actuales, columnas_actuales = entradas.pop(0)
            grupos_a_combinar = []

            # Buscar intersección con los grupos existentes
            for i, (filas, columnas) in enumerate(grupos):
                if filas & filas_actuales or columnas & columnas_actuales:
                    grupos_a_combinar.append(i)

            # Combinar todos los grupos relacionados
            for i in sorted(grupos_a_combinar, reverse=True):  # Combinar desde el último hacia el primero
                filas, columnas = grupos.pop(i)
                filas_actuales |= filas
                columnas_actuales |= columnas

            # Agregar el grupo combinado o nuevo
            grupos.append((filas_actuales, columnas_actuales))

        # Verificar columnas llenas de ceros
        columnas_sin_uso = set(matriz.columns[matriz.sum(axis=0) == 0])
        for columna in columnas_sin_uso:
            grupos.append((set(), {columna}))

        # Convertir los grupos en un diccionario final
        return {f'Grupo_{i+1}': (filas, columnas) for i, (filas, columnas) in enumerate(grupos)}


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

    # metodo que guarda en un archivo la lista de biparticiones
    def guardar_biparticiones(self):
        with open('archivos/biparticiones.txt', 'w') as file:
            for particion in self.__lista_biparticiones:
                file.write(f'{particion}\n')
                file.write('---\n')  # Separador visual entre particiones
   
    def guardar_kparticiones(self):
        with open('archivos/kparticiones.txt', 'w') as file:
            for particion in self.__lista_kparticiones:
                file.write(f'{particion}\n')
                file.write('---\n')
 
