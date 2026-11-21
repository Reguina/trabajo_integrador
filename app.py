"""Aplicación de predicción de churn – Banco Andino.

Ejecutar localmente con:  streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from prediccion import COLUMNAS_ENTRADA, cargar_modelo, leer_bitacora, predecir, validar  # noqa: E402

st.set_page_config(page_title='Predicción de churn – Banco Andino', page_icon='🏦', layout='wide')

PAISES = {'Francia': 'France', 'Alemania': 'Germany', 'España': 'Spain'}
GENEROS = {'Mujer': 'Female', 'Hombre': 'Male'}
SI_NO = {'Sí': 1, 'No': 0}


def miles(n):
    """Formatea un entero con punto como separador de miles (26.169)."""
    return f'{n:,}'.replace(',', '.')


@st.cache_resource
def obtener_modelo():
    return cargar_modelo()


try:
    modelo, info = obtener_modelo()
except Exception as error:  # el caso más común: versión de scikit-learn distinta a la del entrenamiento
    import sklearn
    st.error('No se pudo cargar el modelo. Lo más probable es que la versión de scikit-learn instalada '
             f'({sklearn.__version__}) no sea la misma con la que se entrenó el modelo en Colab. '
             'Revisá la versión en requirements.txt.')
    st.exception(error)
    st.stop()
umbral = info['umbral_contacto']

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header('Modelo en uso')
    st.markdown(f"**Versión:** {info['version']}  \n"
                f"**Algoritmo:** {info['algoritmo']}  \n"
                f"**Entrenado:** {info['fecha_entrenamiento']}")
    st.divider()
    st.markdown('**Desempeño en la evaluación offline**')
    met = info['metricas_evaluacion']
    st.markdown(f"- AUC: {met['AUC']:.3f}\n"
                f"- Contactando al {info['capacidad_contacto']:.0%} de la cartera se alcanza al "
                f"**{met['Captura 20%']:.0%}** de los clientes que se van.")
    st.divider()
    st.markdown(f'**Criterio de contacto:** se recomienda contactar a los clientes con probabilidad '
                f'de churn mayor o igual a **{umbral:.0%}**. Con ese umbral se contacta '
                f'aproximadamente al {info["capacidad_contacto"]:.0%} de la cartera.')

st.title('Predicción de churn')
st.caption('Banco Andino · Equipo de Retención. Estima la probabilidad de que un cliente deje el banco '
           'y recomienda si conviene contactarlo.')

tab_cliente, tab_archivo, tab_bitacora = st.tabs(['Consultar un cliente', 'Cargar un archivo', 'Bitácora'])

# ---------------------------------------------------------------- consulta individual
with tab_cliente:
    with st.form('formulario_cliente'):
        col1, col2, col3 = st.columns(3)
        with col1:
            edad = st.number_input('Edad', min_value=18, max_value=100, value=38, step=1)
            genero = st.selectbox('Género', list(GENEROS))
            pais = st.selectbox('País', list(PAISES))
        with col2:
            antiguedad = st.number_input('Antigüedad como cliente (años)', min_value=0, max_value=10, value=5, step=1)
            productos = st.selectbox('Cantidad de productos', [1, 2, 3, 4], index=1)
            activo = st.radio('¿Es cliente activo?', list(SI_NO), horizontal=True)
            tarjeta = st.radio('¿Tiene tarjeta de crédito?', list(SI_NO), horizontal=True)
        with col3:
            score = st.number_input('Score crediticio', min_value=300, max_value=900, value=650, step=1)
            saldo = st.number_input('Saldo en cuenta', min_value=0.0, max_value=300_000.0, value=0.0, step=1000.0)
            salario = st.number_input('Salario estimado anual', min_value=0.0, max_value=250_000.0,
                                      value=110_000.0, step=1000.0)
        enviado = st.form_submit_button('Calcular probabilidad de churn', type='primary')

    if enviado:
        cliente = pd.DataFrame([{
            'CreditScore': score, 'Geography': PAISES[pais], 'Gender': GENEROS[genero], 'Age': edad,
            'Tenure': antiguedad, 'Balance': saldo, 'NumOfProducts': productos, 'HasCrCard': SI_NO[tarjeta],
            'IsActiveMember': SI_NO[activo], 'EstimatedSalary': salario,
        }])
        st.session_state['resultado_cliente'] = predecir(cliente, modelo, info, origen='app')

    if 'resultado_cliente' in st.session_state:
        r = st.session_state['resultado_cliente'].iloc[0]
        prob = r['probabilidad_churn']
        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric('Probabilidad de churn', f'{prob:.0%}',
                      delta=f'{(prob - info["tasa_churn_entrenamiento"]) * 100:+.0f} puntos vs. promedio de la cartera',
                      delta_color='inverse')
            st.progress(float(prob))
        with col_b:
            if r['contactar'] == 'Sí':
                st.error(f'**Contactar.** La probabilidad supera el umbral de contacto ({umbral:.0%}). '
                         'Se recomienda ofrecerle una acción de retención.')
            else:
                st.success(f'**No es necesario contactar.** La probabilidad está por debajo del umbral '
                           f'de contacto ({umbral:.0%}).')
            st.caption(f'En promedio, el {info["tasa_churn_entrenamiento"]:.0%} de los clientes deja el banco. '
                       'La predicción quedó registrada en la bitácora.')

# ---------------------------------------------------------------- carga de archivo
with tab_archivo:
    st.markdown('Subí un archivo CSV con un cliente por fila. Tiene que incluir estas columnas '
                '(puede tener otras, que se conservan en el resultado):')
    st.code(', '.join(COLUMNAS_ENTRADA), language=None)
    st.caption('Geography: France / Germany / Spain · Gender: Female / Male · HasCrCard e IsActiveMember: 0 o 1.')

    archivo = st.file_uploader('Archivo de clientes', type='csv')
    if archivo is not None:
        clave = f'{archivo.name}-{archivo.size}'
        if st.button('Procesar archivo', type='primary'):
            try:
                datos = pd.read_csv(archivo)
                validos, invalidos = validar(datos)
            except ValueError as error:
                st.error(str(error))
                validos, invalidos = None, None
            if validos is not None and len(validos) > 0:
                with st.spinner(f'Calculando {miles(len(validos))} predicciones...'):
                    resultado = predecir(validos, modelo, info, origen=f'archivo: {archivo.name}')
                resultado = resultado.sort_values('probabilidad_churn', ascending=False)
                resultado.insert(0, 'ranking', range(1, len(resultado) + 1))
                st.session_state['resultado_archivo'] = (clave, archivo.name, resultado, invalidos)
            elif validos is not None:
                st.error('Ningún registro del archivo es válido.')
                st.dataframe(invalidos, width='stretch')

        if st.session_state.get('resultado_archivo', (None,))[0] == clave:
            _, nombre, resultado, invalidos = st.session_state['resultado_archivo']
            a_contactar = (resultado['contactar'] == 'Sí').sum()
            c1, c2, c3 = st.columns(3)
            c1.metric('Clientes procesados', miles(len(resultado)))
            c2.metric('Clientes a contactar', miles(a_contactar))
            c2.caption(f'{a_contactar / len(resultado):.1%} del archivo'.replace('.', ','))
            c3.metric('Probabilidad promedio', f'{resultado["probabilidad_churn"].mean():.1%}'.replace('.', ','))
            if len(invalidos):
                st.warning(f'Se omitieron {miles(len(invalidos))} registros con datos inválidos.')
                with st.expander('Ver registros omitidos'):
                    st.dataframe(invalidos, width='stretch')
            st.markdown('**Clientes ordenados de mayor a menor riesgo** (primeros 100)')
            st.dataframe(resultado.head(100), width='stretch', hide_index=True)
            st.download_button('Descargar todas las predicciones (CSV)',
                               resultado.to_csv(index=False).encode('utf-8'),
                               file_name=f'predicciones_{os.path.splitext(nombre)[0]}.csv', mime='text/csv')

# ---------------------------------------------------------------- bitácora
with tab_bitacora:
    bitacora = leer_bitacora()
    if bitacora.empty:
        st.info('Todavía no hay predicciones registradas.')
    else:
        c1, c2, c3 = st.columns(3)
        ejecuciones = bitacora.groupby('id_ejecucion').first()
        individuales = ejecuciones[ejecuciones['clientes_en_ejecucion'] == 1]
        c1.metric('Predicciones registradas', miles(len(bitacora)))
        c2.metric('Ejecuciones', miles(len(ejecuciones)))
        n_ind, n_arch = len(individuales), len(ejecuciones) - len(individuales)
        c2.caption(f'{miles(n_ind)} {"consulta individual" if n_ind == 1 else "consultas individuales"} y '
                   f'{miles(n_arch)} {"archivo" if n_arch == 1 else "archivos"}')
        if len(individuales):
            c3.metric('Tiempo promedio de una consulta individual',
                      f'{individuales["tiempo_ms_ejecucion"].mean():.0f} ms')
        st.markdown('**Últimas 100 predicciones**')
        st.dataframe(bitacora.tail(100).iloc[::-1], width='stretch', hide_index=True)
        st.download_button('Descargar bitácora completa (CSV)', bitacora.to_csv(index=False).encode('utf-8'),
                           file_name='bitacora.csv', mime='text/csv')
    st.caption('En Streamlit Community Cloud, la bitácora se borra cuando la aplicación se reinicia. '
               'Descargala después de procesar un archivo para conservarla.')
