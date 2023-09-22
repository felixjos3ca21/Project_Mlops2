<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

## Contenido

- [Descripción del problema](#descripción-del-problema)
- [ETL](#etl)
- [Implementación de la API](#implementación-de-la-api)
- [Exploración de los datos (EDA)](#exploración-de-los-datos-eda)
- [Despliegue de la API](#despliegue-de-la-api)
- [Sistema de Recomendación de Películas](#sistema-de-recomendación-de-películas)

<hr>  

## Descripción del problema

## Contexto

En este proyecto, nos encontramos frente a la emocionante tarea de llevar nuestro modelo de recomendación al mundo real. Después de lograr buenas métricas en su rendimiento, surge la pregunta: ¿cómo lo implementamos y mantenemos en producción?

El ciclo de vida de un proyecto de Machine Learning abarca desde la recopilación y procesamiento de los datos (tareas propias de un Ingeniero de Datos) hasta el entrenamiento y mantenimiento continuo del modelo de Machine Learning a medida que llegan nuevos datos.


## Rol a desarrollar

En mi rol como Data Scientist en steam, una plataforma multinacional de videojuegos. Nuestro objetivo principal es desarrollar un sistema de recomendación basado en Machine Learning, el cual aún no ha sido implementado.

Al adentrarme en los datos existentes, me he dado cuenta de que su calidad es deficiente (o incluso inexistente). Los datos están desorganizados, sin transformar, y carecen de procesos automatizados, entre otros problemas. Esta situación dificulta enormemente mi trabajo como Data Scientist.

<hr>  

## ETL
## **Feature Engineering:**

En el conjunto de datos user_reviews, se incluyen reseñas de juegos realizadas por diversos usuarios. Como parte del procesamiento de datos, hemos creado una nueva columna llamada sentiment_analysis. Esta columna se ha generado utilizando análisis de sentimiento mediante NLP (Procesamiento de Lenguaje Natural) y se clasifica en la siguiente escala:

Valor '0' si la reseña es considerada como negativa.
Valor '1' si la reseña es neutral.
Valor '2' si la reseña es positiva.

La introducción de esta nueva columna sentiment_analysis tiene como objetivo simplificar el trabajo con modelos de machine learning y el análisis de datos. En los casos en los que no sea posible realizar el análisis de sentimiento debido a la ausencia de una reseña escrita, la columna tomará automáticamente el valor '1'

<hr>  

## Implementación de la API
<br/>

**`Desarrollo API`**:   Propones disponibilizar los datos de la empresa usando el framework ***FastAPI***. Las consultas que propones son las siguientes:

Se crearon 6 funciones para los endpoints que se consumirán en la API.

+ def **userdata( *`User_id` : str* )**:
    Debe devolver `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

+ def **countreviews( *`YYYY-MM-DD` y `YYYY-MM-DD` : str* )**:
    `Cantidad de usuarios` que realizaron reviews entre las fechas dadas y, el `porcentaje` de recomendación de los mismos en base a reviews.recommend.

+ def **genre( *`género` : str* )**:
    Devuelve el `puesto` en el que se encuentra un género sobre el ranking de los mismos analizado bajo la columna PlayTimeForever. 

+ def **userforgenre( *`género` : str* )**:
    `Top 5` de usuarios con más horas de juego en el género dado, con su URL (del user) y user_id.

+ def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora. 
Ejemplo de salida:
    | Activision ||
    |----------|----------|
    | Año  | Contenido Free  |
    | 2023   | 27% |
    | 2022    | 25%   |
    | xxxx    | xx%   |


+ def **sentiment_analysis( *`año` : int* )**:
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento. 

    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ejemplo de retorno: *{Negative = 182, Neutral = 120, Positive = 278}*

<hr> 

## Exploración de los datos (EDA): _(Exploratory Data Analysis-EDA)_

Después de realizar la limpieza de los datos, me dediqué a explorar las relaciones entre las variables de nuestros conjuntos de datos. Durante este proceso, también busqué posibles valores atípicos o anomalías que podrían resultar interesantes para nuestro análisis, recordando que no todos los datos anómalos son necesariamente errores. Además, estuve atento(a) a la identificación de patrones interesantes que merecieran una exploración más profunda en etapas posteriores.

Para obtener una mejor comprensión de las palabras más frecuentes en los títulos y su posible contribución a nuestro sistema de recomendación, utilicé varias bibliotecas. Utilicé pandas para el análisis y manipulación de los datos, matplotlib.pyplot para generar visualizaciones, sklearn.feature_extraction.text.TfidfVectorizer para extraer características de texto y wordcloud.WordCloud para crear nubes de palabras impactantes.

Estas bibliotecas fueron herramientas valiosas que me permitieron obtener conclusiones significativas sobre nuestros datos. Los gráficos y las nubes de palabras generadas me proporcionaron una visión clara de las palabras más frecuentes y resaltaron los términos clave en los títulos, lo cual puede ser de gran ayuda para nuestro sistema de recomendación.

A lo largo de este proceso de análisis exploratorio, descubrí información relevante que podría influir en el rendimiento y la efectividad de nuestro sistema de recomendación. Estoy emocionado de compartir estos hallazgos y utilizarlos para impulsar nuestro proyecto hacia adelante.
<hr> 

## Despliegue de la API
<br/>

Para poner en marcha nuestro sistema de recomendación de Video Juegos, hemos utilizado la plataforma Render para el despliegue de la API. La API está accesible a través del siguiente enlace: [https://project-mlops2.onrender.com](https://project-mlops2.onrender.com).

Una de las ventajas clave de Render es su facilidad de uso y su capacidad para escalar de manera eficiente. Render se encarga de manejar la infraestructura subyacente y proporciona una plataforma estable y confiable para alojar nuestra API de recomendación de Video Juegos.

Al acceder al enlace de la API, puedes realizar consultas agregando `/docs` al final de la URL. Esto te dirigirá a una interfaz interactiva donde podrás explorar y utilizar los diferentes endpoints disponibles para interactuar con el sistema de recomendación. Desde esta interfaz, podrás ingresar el Id de algun Video Juego y obtener recomendaciones personalizadas.

Render también nos brinda características adicionales, como la capacidad de implementar actualizaciones continuas y automáticas a medida que se agregan nuevos datos y mejoras al modelo. Esto garantiza que nuestro sistema de recomendación esté siempre actualizado y en sintonía con las últimas tendencias y preferencias de los usuarios.

Confiamos en que la combinación de Render como plataforma de despliegue y nuestra potente API de recomendación de Video Juegos brinde una experiencia fluida y atractiva para los usuarios, ofreciendo recomendaciones precisas y relevantes.


<br/>


<hr> 

## Sistema de Recomendación de Películas

Una vez que todos los datos sean consumibles a través de la API y estén listos para ser utilizados por los departamentos de Analytics y Machine Learning, es el momento de entrenar nuestro modelo de machine learning y desarrollar un sistema de recomendación de películas.

El sistema de recomendación se basa en encontrar películas similares a partir de una película de consulta dada. Para lograr esto, utilizamos el algoritmo Nearest Neighbors, que encuentra las películas más similares en función de la similitud de puntuación. Las películas se ordenan según su score de similaridad y se devuelve una lista de las 5 películas más relevantes en orden descendente.

Este algoritmo, junto con el preprocesamiento de los datos, se implementa en la función recomendacion(titulo). Simplemente ingresas el título de una película y obtendrás una lista de las 5 películas más recomendadas.

Los datos se procesaron previamente utilizando el TfidfVectorizer y el NearestNeighbors. Se eliminan caracteres no deseados, se convierte el texto a minúsculas y se eliminan las palabras vacías (stop words) para obtener una representación numérica de las películas.

Para utilizar este sistema de recomendación, asegúrate de que la API esté desplegada correctamente y llama a la función recomendacion(titulo) con el título de la película de consulta. Obtendrás una lista de las películas más relevantes para sugerir a los usuarios.

¡Explora y disfruta de las recomendaciones personalizadas que ofrece nuestro sistema de recomendación de películas!

<br/>
