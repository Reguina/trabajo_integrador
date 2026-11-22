"""Aplicación de predicción de churn – Banco Berlin.

Ejecutar localmente con:  streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

# Las funciones de predicción y bitácora están en src/prediccion.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from prediccion import cargar_modelo, leer_bitacora, predecir, validar  # noqa: E402

# Traducción de lo que elige el usuario a los valores con los que se entrenó el modelo porque originalmente esta en ingles
PAISES = {'Francia': 'France', 'Alemania': 'Germany', 'España': 'Spain'}
GENEROS = {'Mujer': 'Female', 'Hombre': 'Male'}
SI_NO = {'Sí': 1, 'No': 0}


@st.cache_resource  # carga el modelo una sola vez, no en cada clic
def obtener_modelo():
    return cargar_modelo()


modelo, info = obtener_modelo()

st.title('Predicción de churn')
st.caption(f"Modelo versión {info['version']}")

tab_cliente, tab_archivo, tab_bitacora = st.tabs(['Consultar un cliente', 'Cargar un archivo', 'Bitácora'])

# 1. Un cliente
with tab_cliente:
    with st.form('formulario'):  # solo predice al apretar el botón
        edad = st.number_input('Edad', min_value=18, max_value=100, value=38)
        genero = st.selectbox('Género', list(GENEROS))
        pais = st.selectbox('País', list(PAISES))
        antiguedad = st.number_input('Antigüedad como cliente (años)', min_value=0, max_value=10, value=5)
        productos = st.selectbox('Cantidad de productos', [1, 2, 3, 4], index=1)
        activo = st.radio('¿Es cliente activo?', list(SI_NO), horizontal=True)
        tarjeta = st.radio('¿Tiene tarjeta de crédito?', list(SI_NO), horizontal=True)
        score = st.number_input('Score crediticio', min_value=300, max_value=900, value=650)
        saldo = st.number_input('Saldo en cuenta', min_value=0.0, max_value=300_000.0, value=0.0)
        salario = st.number_input('Salario estimado anual', min_value=0.0, max_value=250_000.0, value=110_000.0)
        enviado = st.form_submit_button('Calcular')

    if enviado:
        cliente = pd.DataFrame([{
            'CreditScore': score, 'Geography': PAISES[pais], 'Gender': GENEROS[genero], 'Age': edad,
            'Tenure': antiguedad, 'Balance': saldo, 'NumOfProducts': productos, 'HasCrCard': SI_NO[tarjeta],
            'IsActiveMember': SI_NO[activo], 'EstimatedSalary': salario,
        }])
        resultado = predecir(cliente, modelo, info, origen='app').iloc[0]
        st.metric('Probabilidad de churn', f"{resultado['probabilidad_churn']:.0%}")
        if resultado['contactar'] == 'Sí':
            st.error('Contactar al cliente')
        else:
            st.success('No es necesario contactarlo')

# 2. Un archivo
with tab_archivo:
    archivo = st.file_uploader('Archivo CSV con un cliente por fila', type='csv')
    if archivo is not None and st.button('Procesar archivo'):
        try:
            validos, invalidos = validar(pd.read_csv(archivo))
        except ValueError as error:  # por ejemplo, si al archivo le faltan columnas
            st.error(str(error))
            st.stop()
        resultado = predecir(validos, modelo, info, origen=f'archivo: {archivo.name}')
        resultado = resultado.sort_values('probabilidad_churn', ascending=False)

        st.write(f"Clientes procesados: {len(resultado)} · A contactar: {(resultado['contactar'] == 'Sí').sum()}")
        if len(invalidos):
            st.warning(f'Se omitieron {len(invalidos)} registros con datos inválidos.')
        st.dataframe(resultado)
        st.download_button('Descargar predicciones', resultado.to_csv(index=False), 'predicciones.csv',
                           on_click='ignore')

# 3. Bitácora
with tab_bitacora:
    bitacora = leer_bitacora()
    st.write(f'Predicciones registradas: {len(bitacora)}')
    st.dataframe(bitacora)
    st.download_button('Descargar bitácora', bitacora.to_csv(index=False), 'bitacora.csv', on_click='ignore')
