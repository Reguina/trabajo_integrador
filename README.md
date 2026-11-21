# Predicción de churn bancario

Trabajo práctico integrador – Implementación de modelos de aprendizaje automático (2026).

Modelo que estima la probabilidad de que un cliente deje el banco, para que el equipo de Retención de Banco Andino (organización ficticia) priorice a quién contactar cada mes.

**Aplicación publicada:** _(pegar acá el link de Streamlit Community Cloud)_

## Estructura del repositorio

```
├── app.py                      Aplicación web (Streamlit)
├── src/
│   ├── features.py             Transformaciones del pipeline (punto 2)
│   └── prediccion.py           Carga del modelo, predicción y bitácora (app y monitoreo)
├── modelos/
│   ├── modelo_churn_v1.0.0.joblib   Pipeline completo + modelo entrenado
│   └── modelo_churn_v1.0.0.json     Versión, hiperparámetros, umbral y métricas offline
├── notebooks/
│   ├── 02_preparacion_datos.ipynb   Limpieza, análisis exploratorio y pipeline
│   └── 03_modelado.ipynb            Líneas base, modelos, análisis de errores y selección
├── data/                       Datos (ver "Datos")
├── bitacora/                   Bitácora de predicciones
└── requirements.txt
```

## Datos

Conjunto [Bank Churn Prediction](https://www.kaggle.com/datasets/willianoliveiragibin/bank-churn-prediction) de Kaggle (licencia CC0: Public Domain). Descargar `bank-churn.csv` y guardarlo en `data/`. Pesa 37 MB, así que no se puede subir desde la página web de GitHub (el límite es 25 MB): subirlo con `git` desde la computadora o no incluirlo y dejar solo este enlace.

El notebook 2 genera tres archivos a partir de él:

| Archivo | Uso |
|---|---|
| `data/train.csv` | Entrenamiento (68%) |
| `data/test.csv` | Evaluación offline (17%) |
| `data/produccion.csv` | Lote reservado para simular producción (15%) |

## Cómo ejecutar cada parte

### 1. Notebooks (Google Colab)

Ejecutar en orden `02_preparacion_datos.ipynb` y `03_modelado.ipynb` (el punto 3 continúa en la misma sesión del punto 2). El notebook 3 guarda el modelo en `modelos/`. Descargar los dos archivos de esa carpeta y subirlos al repositorio.

### 2. Aplicación en una computadora local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador, en `http://localhost:8501`.

### 3. Aplicación publicada (Streamlit Community Cloud)

1. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
2. **Create app** → elegir este repositorio, la rama `main` y el archivo `app.py`.
3. En **Advanced settings**, elegir la misma versión de Python que usa Colab (o la más cercana disponible).
4. **Deploy**. Cada cambio que se sube al repositorio actualiza la app automáticamente.

## La aplicación

| Pestaña | Qué hace |
|---|---|
| Consultar un cliente | Formulario con los 10 datos del cliente. Devuelve la probabilidad de churn y si conviene contactarlo. |
| Cargar un archivo | Recibe un CSV con muchos clientes y devuelve las predicciones ordenadas de mayor a menor riesgo, para descargar. Omite los registros con datos inválidos e informa cuántos fueron. |
| Bitácora | Muestra todas las predicciones registradas y permite descargar la bitácora. |

Un cliente se marca para **contactar** cuando su probabilidad supera el umbral guardado en el `.json` del modelo (0,38), que corresponde a contactar aproximadamente al 20% de la cartera.

## Bitácora

Cada predicción, individual o por archivo, agrega una fila a `bitacora/bitacora.csv` con:

| Columna | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora de la predicción |
| `version_modelo` | Versión del modelo que la hizo |
| `origen` | `app` (consulta individual) o `archivo: <nombre>` |
| `id_ejecucion` | Identifica a todas las predicciones de una misma consulta o archivo |
| 10 columnas de entrada | Datos del cliente tal como se recibieron |
| `probabilidad_churn`, `contactar` | Predicción |
| `tiempo_ms_por_cliente`, `tiempo_ms_ejecucion`, `clientes_en_ejecucion` | Tiempo de ejecución |

Se guardan los datos de entrada completos para poder analizar desvíos en el monitoreo. En Streamlit Community Cloud el almacenamiento no es permanente: la bitácora se borra cuando la app se reinicia, por lo que hay que descargarla desde la pestaña **Bitácora** después de procesar un archivo.
