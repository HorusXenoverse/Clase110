# Importar la biblioteca OpenCV
from contextlib import AsyncExitStack
from statistics import mode
import cv2
import numpy as np
#Siempre se debe instalar el tensorflow en donde se use
import tensorflow as tf

#Importamos el modelo de la cara, siempre hay q poner toda la ruta en donde se encuentre
model = tf.keras.models.load_model("D:/Proyectos de la escuela/Quinta etapa/Clase 110/keras_model.h5")

# Definir un objeto de captura de video
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capturar el video fotograma por fotograma
    ret, frame = vid.read()

    #Hace q coincidan los tamaños entre la webCam y el modelo
    image = cv2.resize(frame, (224,224))

    #Para concertir en una matriz de 3 dimensiones la imagen
    imageMatriz = np.array(image, dtype = np.float32)

    #Convierte la matriz de 3 dimensiones en 4 ya q la normalización trabaja con matrices de 4 dimensiones
    convertirMatriz = np.expand_dims(imageMatriz, axis = 0)

    #Dividimos entre 255 para q cada pixel de la matriz se guarde con un rango de entre 0 a 1
    normalizeImage = convertirMatriz / 255.0
    
    #La utilizamos para utilizar el modelo y q nos diga cuanto porcentaje está segura
    predictionModel = model.predict(normalizeImage)
    print("Prediccion", predictionModel)
  
    # Mostrar el fotograma resultante
    cv2.imshow('Fotograma', frame)
      
    # Salir de la ventana con la barra espaciadora
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Después del bucle, liberar al objeto de captura
vid.release()

# Destruir todas las ventanas
cv2.destroyAllWindows()
