import csv
import json

class Sistema:

    def __init__(self, ruta):
        self.__estado_inicial = None
        self.__sistema_candidato = None
        self.__subsistema_futuro = None
        self.__subsistema_presente = None
        self.set_with_csv(ruta)

    """
    Lee un archivo CSV y establece los atributos del objeto con los valores de la primera fila.

    Parámetros:
    ruta (str): La ruta del archivo CSV a leer.
    """
    def set_with_csv(self, ruta):
        with open(ruta, mode='r') as archivo:
            lector = csv.reader(archivo)
            fila = next(lector)
            if len(fila) >= 4:
                self.__estado_inicial = fila[0].strip()
                self.__sistema_candidato = fila[1].strip()
                self.__subsistema_futuro = fila[2].strip()
                self.__subsistema_presente = fila[3].strip()
            else:
                raise ValueError("El archivo CSV no tiene suficientes columnas")
            
    def set_with_json(self, ruta):
        with open(ruta, mode='r') as archivo:
            contenido = json.load(archivo)
            self.__estado_inicial = contenido["estado_inicial"]
            self.__sistema_candidato = contenido["background"]
            self.__subsistema_futuro = contenido["subsistema_futuro"]
            self.__subsistema_presente = contenido["subsistema_presente"]

    def get_estado_inicial(self):
        return self.__estado_inicial
    
    def get_sistema_candidato(self):
        return self.__sistema_candidato
    
    def get_subsistema_presente(self):
        return self.__subsistema_presente
    
    def get_subsistema_futuro(self):
        return self.__subsistema_futuro

    def set_estado_inicial(self, estado_inicial):
        self.__estado_inicial = estado_inicial

    def set_sistema_candidato(self, background):
        self.__sistema_candidato = background
    
    def set_subsistema_presente(self, subsistema_presente):
        self.__subsistema_presente = subsistema_presente

    def set_subsistema_futuro(self, subsistema_futuro):
        self.__subsistema_futuro = subsistema_futuro

    def __repr__(self):
        return (f"Estado_inicial={self.__estado_inicial}, "
                f"background={self.__sistema_candidato}, subsistema_presente={self.__subsistema_presente}, "
                f"subsistema_futuro={self.__subsistema_futuro})")