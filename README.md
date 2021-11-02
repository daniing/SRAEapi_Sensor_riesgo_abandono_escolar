# SRAEapi_risk_school_dropout
Aquí se encuentra la información de una aplicación para apoyar la detección del riesgo de deserción escolar.

Esta aplicación fue el resultado de una investigación desarrollada en la línea de inteligencia artificial, de acuerdo al despliegue del proyecto de minería de datos implementado en la IEBS (Institucion Educativa Barrio Santander).

Los modelos fueron entrenados con cuatro tipos de características:
      - Características demográficas
      - Transformación de características demográficas: Métricas de caratcerísticas demográficas
      - Características académicas
      - Transformación de características académicas: Métricas de caratcerísticas académicas
      
El esquema de detección se planteo utilizando un modelo de "latefusion". Inicialmente cuatro clasificadores expertos en cada tipo de características y luego se obtenia la probabilidad de cada uno de los clasificadores a la entrada de un clasificador final.

Características estudiantiles:
Demográficas
  - Fechas de Nacimiento
  - Número de hermanos en el colegio
  - Género
  - Grado actual: grado al cual el estudiante se matricula en el año de referencia o en el cual se va a realizar la predicción del riesgo de deserción.
  - Puntaje Sisbén: la investigación inicio en el año 2020 aún se contaba con el puntaje del sisbén III
  - Distancia (kilómetros): esta medida debe registrarse porque no se cuenta con conexión a la API de google (georreferenciación a partir de direcciones) para convertir direcciones a coordenadas de latitud y longitud (y así calcular la distancia a la IEBS con la formula de Haversine)
  - Año de ingreso: indica el año cuando el estudiante por primera vez se matriculó en la IEBS
  - Grado de ingreso: indica el grado cuando el estudiante por primera vez se matriculó en la IEBS

Académicas (notas - calificaciones):
  - Matemáticas 
  - Ciencias Naturales 
  - Inglés 
  - Español 
  - Cátedra de Paz 
  - Sociales 
  - Religión 
  - Artes 
  - Educación Física 
  - Tecnología 
  - Emprendimiento 
  - Ética y Valores
