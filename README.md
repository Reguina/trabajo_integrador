Predicción de churn bancario

Trabajo práctico integrador de la materia Implementación de modelos de aprendizaje automático (2026).

En este proyecto desarrollamos un modelo que estima la probabilidad de que un cliente deje el banco. La idea es que el equipo de Retención de Banco Berlin (una organización ficticia) pueda usarlo cada mes para decidir a qué clientes contactar primero, antes de que se vayan.

La aplicación está publicada en https://trabajointegrador-implementacion-uba.streamlit.app/

Qué hay en el repositorio

El repositorio está organizado en varias carpetas. En la carpeta principal está el archivo app.py, que contiene la aplicación web hecha con Streamlit, y el archivo requirements.txt, con las librerías necesarias para ejecutarla.

La carpeta src guarda el código que comparten la notebook y la aplicación. El archivo features.py contiene las transformaciones del pipeline, es decir, la limpieza y la construcción de las variables que usa el modelo. El archivo prediccion.py se encarga de cargar el modelo, validar los datos de los clientes, hacer las predicciones y registrarlas en la bitácora.

En la carpeta modelos está el modelo elegido, un Gradient Boosting, guardado en dos archivos: modelo_churn_v1.0.0.joblib, que contiene el pipeline completo junto con el modelo entrenado, y modelo_churn_v1.0.0.json, que registra la versión, los hiperparámetros, el umbral de contacto y las métricas obtenidas en la evaluación. Tambien se habia probado el random forest.

La carpeta notebooks contiene la notebook Trabajo-practico-integrados.ipynb, donde se hace la limpieza de los datos, el análisis exploratorio, la construcción del pipeline, el entrenamiento de los modelos y la selección del modelo final.

En la carpeta data están el archivo produccion.csv, que es el lote de clientes que reservamos para simular la puesta en producción, y el archivo ejemplo_clientes.csv, un archivo chico para probar la aplicación. Por último, la carpeta bitacora es donde la aplicación guarda el registro de las predicciones.

Los datos

Trabajamos con el conjunto Bank Churn Prediction de Kaggle (https://www.kaggle.com/datasets/willianoliveiragibin/bank-churn-prediction), que tiene información de 175.028 clientes y se publica con licencia CC0, es decir, de dominio público. El archivo original, bank-churn.csv, pesa 37 MB y no lo incluimos en el repositorio porque supera el límite de 25 MB que permite la página web de GitHub, pero se puede descargar desde ese enlace.

A partir de ese archivo, la notebook separa a los clientes en tres grupos. El primero, con 118.630 clientes (el 68% del total), se usa para entrenar los modelos. El segundo, con 29.658 clientes (el 17%), se usa para evaluarlos y elegir el mejor. El tercero, con 26.169 clientes (el 15% restante), se reserva desde el principio para simular a los clientes nuevos que llegarían una vez publicado el modelo. 

Cómo ejecutar el proyecto
La notebook

La notebook está pensada para ejecutarse en Google Colab. Para hacerlo, hay que entrar a Colab, subir la notebook desde el menú Archivo --> Subir notebook y subir también el archivo bank-churn.csv a la carpeta principal de Colab (/content/), que es donde la notebook lo busca. Después alcanza con ir a Entorno de ejecución --> Ejecutar todo. Al terminar, la notebook guarda el modelo en la carpeta modelos de Colab, y esos dos archivos son los que hay que subir a la carpeta modelos del repositorio.

Entrenamos el modelo con Python 3.13 y scikit-learn 1.6.1, que eran las versiones de Colab en ese momento. Es importante que requirements.txt indique la misma versión de scikit-learn, porque un modelo guardado con una versión puede no abrir con otra.


La aplicación publicada

La publicamos de forma gratuita en Streamlit Community Cloud. Para hacerlo, hay que entrar a share.streamlit.io con la cuenta de GitHub, elegir Create app, seleccionar este repositorio, la rama main y el archivo app.py, y en Advanced settings elegir Python 3.13. Después de hacer clic en Deploy, la aplicación queda en línea, y cada cambio que se sube al repositorio la actualiza automáticamente.

Cómo se usa la aplicación

La aplicación tiene tres pestañas. En la primera, Consultar un cliente, se cargan los diez datos de un cliente en un formulario y la aplicación muestra su probabilidad de dejar el banco y si conviene contactarlo.

En la segunda, Cargar un archivo, se sube un archivo CSV con muchos clientes y la aplicación devuelve las predicciones de todos, ordenadas de mayor a menor riesgo y listas para descargar. Si algún cliente tiene datos inválidos, por ejemplo una edad imposible o un país que el modelo no conoce, la aplicación lo deja afuera y avisa cuántos fueron. Para probar esta pestaña se puede usar el archivo data/ejemplo_clientes.csv.

El archivo que se sube tiene que incluir las columnas CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember y EstimatedSalary. En Geography se acepta France, Germany o Spain, y en Gender, Female o Male. Si el archivo tiene otras columnas, como un número de cliente, se conservan en el resultado.

En la tercera pestaña, Bitácora, se pueden ver todas las predicciones que se hicieron y descargarlas.

La aplicación recomienda contactar a un cliente cuando su probabilidad de irse es del 38% o más. Elegimos ese umbral porque, con él, se contacta aproximadamente al 20% de la cartera, que es la capacidad que suponemos que tiene el equipo de Retención. El valor está guardado en el archivo .json del modelo, así que si en el futuro se reentrena, el nuevo modelo trae su propio umbral.

La bitácora

Cada vez que la aplicación hace una predicción, ya sea de un cliente o de un archivo completo, agrega una fila por cliente al archivo bitacora/bitacora.csv. En cada fila queda registrado cuándo se hizo la predicción, qué versión del modelo la hizo y si vino del formulario o de un archivo, junto con un código que agrupa todas las predicciones de una misma consulta o archivo. También se guardan los diez datos del cliente tal como llegaron a la aplicación, la probabilidad que calculó el modelo, la recomendación de contactarlo o no y cuánto tiempo tardó la predicción.

Guardamos los datos completos de cada cliente, y no solo un resumen, porque los necesitamos en el monitoreo para comparar a los clientes nuevos con los del entrenamiento.

Hay un detalle a tener en cuenta: en Streamlit Community Cloud los archivos no se guardan de forma permanente, así que la bitácora se borra cada vez que la aplicación se reinicia. Por eso, después de procesar un archivo, conviene descargarla desde la pestaña Bitácora.