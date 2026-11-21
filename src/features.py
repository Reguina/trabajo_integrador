"""Transformaciones del modelo de churn. Las usan el entrenamiento, la app y el script por lotes."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Columnas que debe traer cada cliente (así llegan desde los sistemas del banco o la app)
COLUMNAS_ENTRADA = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
OBJETIVO = 'Exited'

NUMERICAS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
             'SaldoSobreSalario', 'AntiguedadSobreEdad', 'EdadInactivo']
BINARIAS = ['HasCrCard', 'IsActiveMember', 'SaldoCero']
CATEGORICAS = ['Geography', 'Gender', 'Productos']


def construir_variables(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables construidas a partir de las columnas originales."""
    X = X[COLUMNAS_ENTRADA].copy()
    X['SaldoCero'] = (X['Balance'] == 0).astype(int)
    X['SaldoSobreSalario'] = X['Balance'] / X['EstimatedSalary'].clip(lower=1)
    X['AntiguedadSobreEdad'] = X['Tenure'] / X['Age']
    X['EdadInactivo'] = X['Age'] * (1 - X['IsActiveMember'])
    X['Productos'] = np.where(X['NumOfProducts'] >= 3, '3+', X['NumOfProducts'].astype(int).astype(str))
    return X


def nombres_variables(transformador, columnas_entrada):
    """Nombres de las columnas que devuelve construir_variables (los usa scikit-learn)."""
    return np.array(COLUMNAS_ENTRADA + ['SaldoCero', 'SaldoSobreSalario', 'AntiguedadSobreEdad',
                                        'EdadInactivo', 'Productos'])


def crear_preprocesamiento() -> Pipeline:
    """Pipeline de transformaciones: construcción de variables + imputación + escalado + codificación."""
    columnas = ColumnTransformer([
        ('num', Pipeline([('imputar', SimpleImputer(strategy='median')),
                          ('escalar', StandardScaler())]), NUMERICAS),
        ('bin', SimpleImputer(strategy='most_frequent'), BINARIAS),
        ('cat', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                          ('codificar', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAS),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ('variables', FunctionTransformer(construir_variables, feature_names_out=nombres_variables)),
        ('columnas', columnas),
    ])
