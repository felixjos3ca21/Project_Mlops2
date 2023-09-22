from fastapi import FastAPI
import pandas as pd
import math
import numpy as np
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ast


app = FastAPI()
#--------------------------------------------
@app.get("/")
def index():
    return {"Mensaje" : "Hola Mundo"}
#--------------------------------------------

#-----------------------------------------------------------------------------------------------------------------
@app.get("/userdata/")
async def userdata(user_id: str):
    point_one = pd.read_csv('point_one.csv')
    point_one['user_id'] = point_one['user_id'].astype(str)
    user_id = str(user_id)
    # Filtrar el DataFrame por el user_id proporcionado
    usuario = point_one[point_one['user_id'] == user_id]

    if usuario.empty:
        return {"error": "Usuario no encontrado"}

    # Obtener la información del usuario
    dinero_gastado = usuario['gasto_user'].values[0].item()
    porcentaje_recomendacion = usuario['recommend_percentage'].values[0].item()
    cantidad_items = usuario['items_count'].values[0].item()

    # Crear el diccionario de respuestas
    respuesta = {
        "Cantidad de Dinero Gastado": dinero_gastado,
        "Porcentaje de Recomendacion": porcentaje_recomendacion,
        "Cantidad de Items": cantidad_items
    }

    return respuesta

#-----------------------------------------------------------------------------------------------------------------

@app.get("/countreviews/")
async def countreviews(fecha_inicio: str, fecha_fin: str):
    point_second = pd.read_csv('point_second.csv')
    # Convierte las fechas de entrada a objetos datetime
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    # Utiliza .loc para convertir la columna 'posted' a objetos datetime
    point_second.loc[:, 'posted'] = pd.to_datetime(point_second['posted'])

    # Filtra el DataFrame para incluir solo las filas dentro del rango de fechas
    point_second_filtrado = point_second[(point_second['posted'] >= fecha_inicio) & (point_second['posted'] <= fecha_fin)]

    # Cantidad de usuarios con reviews en el rango de fechas
    cantidad_usuarios_reviews = point_second_filtrado['user_id'].nunique()

    # Porcentaje de recomendación en el rango de fechas
    porcentaje_recomendacion = ((point_second_filtrado['recommend'].mean()) * 100).round(2)

    # Crea un diccionario con las respuestas
    respuesta = {
        'Cantidad de usuarios con Reviews': cantidad_usuarios_reviews,
        'Porcentaje de Recomendación': porcentaje_recomendacion
    }

    return respuesta

#-----------------------------------------------------------------------------------------------------------------

@app.get("/genre/")
async def genre(genero: str):
    point_thirth = pd.read_csv('point_thirth.csv')
    # Convierte el género consultado a minúsculas para hacer la búsqueda insensible a mayúsculas y minúsculas
    genero = genero.lower()

    try:
        # Realiza la búsqueda en los datos para encontrar la posición del género
        posicion = point_thirth.index[point_thirth['genre'].str.lower() == genero][0] + 1  # Suma 1 para comenzar desde 1 en lugar de 0
        respuesta = f"El género '{genero}' se encuentra en la posición: {posicion}"
    except IndexError:
        respuesta = f"El género '{genero}' no se encuentra en el ranking."

    return {"respuesta": respuesta}

#-----------------------------------------------------------------------------------------------------------------

@app.get("/userforgenre/")
async def userforgenre(genero: str):
    point_fourth = pd.read_csv('point_fourth.csv')
    # Convierte el género consultado a minúsculas
    genero = genero.lower()
    point_fourth['genres'] = point_fourth['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Filtra el DataFrame para obtener solo las filas que contienen el género consultado
    df_filtrado = point_fourth[point_fourth['genres'].apply(lambda x: genero in [g.lower() for g in x] if isinstance(x, list) else False)]

    if df_filtrado.empty:
        return {"mensaje": f"No se encontraron registros para el género '{genero}'."}

    # Agrupar por user_id y calcular la suma de playtime_forever
    ranking = df_filtrado.groupby('user_id', as_index=False)['playtime_forever'].sum()

    # Ordenar en orden descendente y tomar los 5 mejores resultados
    ranking = ranking.sort_values(by='playtime_forever', ascending=False).head(5)

    # Obtener las URLs de usuario para los usuarios en el top 5
    urls = df_filtrado[df_filtrado['user_id'].isin(ranking['user_id'])][['user_id', 'user_url']].drop_duplicates()

    # Fusionar los datos de ranking y URLs
    top5_usuarios = pd.merge(ranking, urls, on='user_id', how='left')

    # Convertir el resultado a un diccionario
    resultado = {
        f"Para el género '{genero}'": top5_usuarios.to_dict(orient='records')
    }

    return resultado

#-----------------------------------------------------------------------------------------------------------------

@app.get("/developer/")
async def developer(developer: str):
    point_fiveth = pd.read_csv('point_fiveth.csv')
    developer = developer.lower()
    
    # Filtrar el DataFrame 'point_five' para obtener solo las filas con el desarrollador especificado
    desarrollador_filtrado = point_fiveth[point_fiveth['developer'].str.lower() == developer]

    if desarrollador_filtrado.empty:
        return {"mensaje": f"No se encontraron registros para el desarrollador '{developer}'."}

    # Calcular la cantidad de veces que aparece el desarrollador
    cantidad_items = len(desarrollador_filtrado)

    # Calcular el porcentaje de juegos gratuitos ('Free') por año
    juegos_gratuitos_por_anio = desarrollador_filtrado[desarrollador_filtrado['price'] == 'Free'].groupby('release_year').size()
    total_juegos_por_anio = desarrollador_filtrado.groupby('release_year').size()
    porcentaje_gratuitos_por_anio = ((juegos_gratuitos_por_anio / total_juegos_por_anio) * 100).round(2)
    
    # Reemplazar NaN con None en el porcentaje
    porcentaje_gratuitos_por_anio = porcentaje_gratuitos_por_anio.replace({np.nan: 0})

    # Obtener el nombre del editor (publisher) del primer juego del desarrollador
    publisher = desarrollador_filtrado['publisher'].iloc[0]

    # Crear un diccionario con los resultados
    resultados = {
            'Publisher': publisher,
            'Cantidad de Items': cantidad_items,
            'Porcentaje de Free por Año': dict(porcentaje_gratuitos_por_anio)}
    
    return resultados

#-----------------------------------------------------------------------------------------------------------------

@app.get("/sentiment_analysis/")
async def sentiment_analysis(year: int):
    
    point_sixth = pd.read_csv('point_sixth.csv')
    # Aquí debes agregar tu código para cargar y procesar el DataFrame point_sixth

    # Filtra las filas del DataFrame para el año dado
    df_filtrado = point_sixth[point_sixth['release_date'] == year]

    # Cuenta el número de ocurrencias de cada sentimiento para el año dado
    conteo_sentimientos = df_filtrado['sentiment_analysis'].value_counts().to_dict()

    # Mapea los valores numéricos a etiquetas de sentimientos
    mapeo_sentimientos = {0: 'Negativos', 1: 'Neutrales', 2: 'Positivos'}

    # Crea el diccionario final con etiquetas de sentimientos y conteos
    diccionario_resultado = {mapeo_sentimientos[sentimiento]: conteo for sentimiento, conteo in conteo_sentimientos.items()}

    return {"Las reseñas del año consultado": diccionario_resultado}

#-----------------------------------------------------------------------------------------------------------------

@app.get('/recomendacion_juego/{id}')
def recomendacion_juego(id: str):
    try:
        '''Ingresa un ID de juego y obtén recomendaciones similares en una lista'''
        data_ml = pd.read_csv('data_ML.csv')
         # Crear un objeto TfidfVectorizer para convertir el texto en vectores TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        # Aplicar el vectorizador a los datos de texto combinados y obtener la matriz de vectores TF-IDF
        vectorized_data = vectorizer.fit_transform(data_ml['combined_text'])
        
        # Crear y ajustar el modelo KNN fuera de la función
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(vectorized_data)
        data_ml['id'] = data_ml['id'].astype(str)
        # Verificar si el ID existe en el DataFrame
        if id not in data_ml['id'].values:
            raise HTTPException(status_code=404, detail="ID de juego no encontrado")
        
        # Obtener índice del ID
        index = data_ml[data_ml['id'] == id].index[0]
        
        # Numero de recomendaciones
        num_recomen = 5
        
        # Obtener recomendaciones basadas en el índice de consulta
        _, indices = knn_model.kneighbors(vectorized_data[index], n_neighbors=num_recomen+1)
        
        # Obtener índices de los juegos recomendados
        index_game = indices.flatten()[1:]
        
        # Devolver una estructura de diccionario con los IDs de los juegos recomendados
        result = {'lista recomendada': data_ml['title'].iloc[index_game].tolist()}
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")