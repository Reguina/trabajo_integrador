# Predicción de churn bancario

Trabajo práctico integrador – Implementación de modelos de aprendizaje automático (2026).

Modelo que estima la probabilidad de que un cliente deje el banco, para que el equipo de Retención de Banco Berlin (organización ficticia) priorice a quién contactar cada mes.

**Aplicación publicada:** https://trabajointegrador-implementacion-uba.streamlit.app/

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
│   ├── Trabajo-practico-integrados.ipynb   Limpieza, análisis exploratorio, pipeline, seleccion modelo
├── data/                       Datos (ver "Datos")
├── bitacora/                   Bitácora de predicciones
└── requirements.txt
```

Datos

Conjunto Bank Churn Prediction de Kaggle (licencia CC0: Public Domain), con 175.028 clientes. El archivo original (bank-churn.csv, 37 MB) no está en el repositorio porque supera el límite de 25 MB de la página web de GitHub: se descarga desde el enlace.

La notebook genera tres archivos a partir de él:

Archivo	Clientes	Uso	¿Está en el repositorio?
data/train.csv	118.630 (68%)	Entrenamiento	No: se genera al ejecutar la notebook
data/test.csv	29.658 (17%)	Evaluación offline	No: se genera al ejecutar la notebook
data/produccion.csv	26.169 (15%)	Lote reservado para simular producción	Sí
Cómo ejecutar cada parte
1. Notebook (Google Colab)
Abrir notebooks/Trabajo_practico_integrados.ipynb en Google Colab (Archivo → Subir notebook).
Subir bank-churn.csv a la carpeta principal de Colab (/content/), que es desde donde lo lee la notebook.
Entorno de ejecución → Ejecutar todo. 
Al terminar, la notebook guarda el modelo en la carpeta modelos/ de Colab. Descargar los dos archivos y subirlos a modelos/ en el repositorio.

Versiones usadas: Python 3.13 y scikit-learn 1.6.1 (las de Colab al momento del entrenamiento). La versión de scikit-learn de requirements.txt tiene que coincidir con la del entrenamiento; si no, el modelo puede no cargar.

2. Aplicación en una computadora local
bash
pip install -r requirements.txt
streamlit run app.py

Se abre en el navegador, en http://localhost:8501.

3. Aplicación publicada (Streamlit Community Cloud)
Entrar a share.streamlit.io con la cuenta de GitHub.
Create app → elegir este repositorio, la rama main y el archivo app.py.
En Advanced settings, elegir Python 3.13.
Deploy. Cada cambio que se sube al repositorio actualiza la app automáticamente.
La aplicación
Pestaña	Qué hace
Consultar un cliente	Formulario con los 10 datos del cliente. Devuelve la probabilidad de churn y si conviene contactarlo.
Cargar un archivo	Recibe un CSV con muchos clientes y devuelve las predicciones ordenadas de mayor a menor riesgo, para descargar. Omite los registros con datos inválidos e informa cuántos fueron. Para probarla, usar data/ejemplo_clientes.csv.
Bitácora	Muestra todas las predicciones registradas y permite descargar la bitácora.

El archivo que se carga tiene que tener estas columnas (puede tener otras, que se conservan en el resultado): CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary. Geography acepta France, Germany o Spain; Gender, Female o Male.

Un cliente se marca para contactar cuando su probabilidad es mayor o igual al umbral guardado en el .json del modelo (0,38), que corresponde a contactar aproximadamente al 20% de la cartera.

Bitácora

Cada predicción, individual o por archivo, agrega una fila a bitacora/bitacora.csv con:

Columna	Contenido
fecha_hora	Fecha y hora de la predicción
version_modelo	Versión del modelo que la hizo
origen	app (consulta individual) o archivo: <nombre>
id_ejecucion	Identifica a todas las predicciones de una misma consulta o archivo
10 columnas de entrada	Datos del cliente tal como se recibieron
probabilidad_churn, contactar	Predicción
tiempo_ms_por_cliente, tiempo_ms_ejecucion, clientes_en_ejecucion	Tiempo de ejecución

Se guardan los datos de entrada completos para poder analizar desvíos en el monitoreo. En Streamlit Community Cloud el almacenamiento no es permanente: la bitácora se borra cuando la app se reinicia, por lo que hay que descargarla desde la pestaña Bitácora después de procesar un archivo.
