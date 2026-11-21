"""Predicción con el modelo de churn publicado y registro en la bitácora.

Lo usan la aplicación (app.py) y el monitoreo del punto 5, para que todas las
predicciones se hagan y se registren exactamente de la misma manera.
"""
import glob
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# El modelo se guardó importando `features` desde src/, así que tiene que poder importarse igual al cargarlo
CARPETA_SRC = os.path.dirname(os.path.abspath(__file__))
if CARPETA_SRC not in sys.path:
    sys.path.insert(0, CARPETA_SRC)

RAIZ = os.path.dirname(CARPETA_SRC)
CARPETA_MODELOS = os.path.join(RAIZ, 'modelos')
RUTA_BITACORA = os.path.join(RAIZ, 'bitacora', 'bitacora.csv')

COLUMNAS_ENTRADA = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Valores aceptados para cada columna (los mismos rangos que aparecen en el entrenamiento)
RANGOS = {
    'CreditScore': (300, 900),
    'Age': (18, 100),
    'Tenure': (0, 10),
    'Balance': (0, 300_000),
    'NumOfProducts': (1, 4),
    'HasCrCard': (0, 1),
    'IsActiveMember': (0, 1),
    'EstimatedSalary': (0, 250_000),
}
CATEGORIAS = {
    'Geography': ['France', 'Germany', 'Spain'],
    'Gender': ['Female', 'Male'],
}


def version_mas_reciente():
    """Devuelve la versión más nueva que hay en la carpeta modelos/ (por ejemplo, '1.0.0')."""
    archivos = glob.glob(os.path.join(CARPETA_MODELOS, 'modelo_churn_v*.joblib'))
    if not archivos:
        raise FileNotFoundError('No hay ningún modelo en la carpeta modelos/.')
    versiones = [os.path.basename(a)[len('modelo_churn_v'):-len('.joblib')] for a in archivos]
    return max(versiones, key=lambda v: tuple(int(p) for p in v.split('.')))


def cargar_modelo(version=None):
    """Carga el pipeline del modelo y su archivo de información. Sin versión, usa la más reciente."""
    version = version or version_mas_reciente()
    modelo = joblib.load(os.path.join(CARPETA_MODELOS, f'modelo_churn_v{version}.joblib'))
    with open(os.path.join(CARPETA_MODELOS, f'modelo_churn_v{version}.json'), encoding='utf-8') as f:
        info = json.load(f)
    return modelo, info


def validar(datos: pd.DataFrame):
    """Separa los registros válidos de los inválidos.

    Devuelve (validos, invalidos). `invalidos` tiene una columna 'motivo' que explica el problema.
    Si faltan columnas, lanza un ValueError porque no se puede predecir ningún registro.
    """
    faltantes = [c for c in COLUMNAS_ENTRADA if c not in datos.columns]
    if faltantes:
        raise ValueError(f'Faltan columnas en el archivo: {", ".join(faltantes)}')

    datos = datos.copy()
    motivos = pd.Series('', index=datos.index)
    for col, (minimo, maximo) in RANGOS.items():
        valores = pd.to_numeric(datos[col], errors='coerce')
        malo = valores.isna() | (valores < minimo) | (valores > maximo)
        motivos[malo] += f'{col} fuera de rango o vacío; '
        datos[col] = valores
    for col, opciones in CATEGORIAS.items():
        malo = ~datos[col].isin(opciones)
        motivos[malo] += f'{col} debe ser {"/".join(opciones)}; '

    invalidos = datos[motivos != ''].assign(motivo=motivos[motivos != ''].str.rstrip('; '))
    return datos[motivos == ''], invalidos


def predecir(datos: pd.DataFrame, modelo, info, origen='app', registrar=True, ruta_bitacora=RUTA_BITACORA):
    """Predice la probabilidad de churn de cada cliente y registra cada predicción en la bitácora.

    `datos` debe tener las columnas de COLUMNAS_ENTRADA (puede tener otras, que se conservan).
    Devuelve una copia de `datos` con las columnas 'probabilidad_churn' y 'contactar'.
    """
    inicio = time.perf_counter()
    probabilidad = modelo.predict_proba(datos[COLUMNAS_ENTRADA])[:, 1]
    duracion_ms = (time.perf_counter() - inicio) * 1000

    resultado = datos.copy()
    resultado['probabilidad_churn'] = probabilidad.round(4)
    resultado['contactar'] = np.where(probabilidad >= info['umbral_contacto'], 'Sí', 'No')

    if registrar:
        ahora = datetime.now()
        bitacora = pd.DataFrame({
            'fecha_hora': ahora.strftime('%Y-%m-%d %H:%M:%S'),
            'version_modelo': info['version'],
            'origen': origen,
            'id_ejecucion': ahora.strftime('%Y%m%d-%H%M%S-%f'),
            **{col: datos[col].to_numpy() for col in COLUMNAS_ENTRADA},
            'probabilidad_churn': resultado['probabilidad_churn'].to_numpy(),
            'contactar': resultado['contactar'].to_numpy(),
            'tiempo_ms_por_cliente': round(duracion_ms / len(datos), 3),
            'tiempo_ms_ejecucion': round(duracion_ms, 1),
            'clientes_en_ejecucion': len(datos),
        })
        os.makedirs(os.path.dirname(ruta_bitacora), exist_ok=True)
        bitacora.to_csv(ruta_bitacora, mode='a', index=False, header=not os.path.exists(ruta_bitacora))

    return resultado


def leer_bitacora(ruta_bitacora=RUTA_BITACORA):
    """Devuelve la bitácora completa (vacía si todavía no hay predicciones)."""
    if not os.path.exists(ruta_bitacora):
        return pd.DataFrame()
    return pd.read_csv(ruta_bitacora, dtype={'id_ejecucion': str})
