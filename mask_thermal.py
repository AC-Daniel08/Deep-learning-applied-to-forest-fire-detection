import cv2
import numpy as np
 
#Cargamos la imagen:
img = cv2.imread("/media/Datos/Escritorio/Documentos tesis/Documento_final_tesis/base de datos termicas/base_datos_termicas_final/incendios/Ter114.jpg")
#Función auxiliar:
def nothing(x):
   pass
 
#Creamos la ventana con las trackbars:
cv2.namedWindow('Parametros1')
cv2.createTrackbar('Blue Minimo','Parametros1',0,255,nothing)
cv2.createTrackbar('Blue Maximo','Parametros1',0,255,nothing)
cv2.createTrackbar('Green Minimo','Parametros1',0,255,nothing)
cv2.createTrackbar('Green Maximo','Parametros1',0,255,nothing)
cv2.createTrackbar('Red Minimo','Parametros1',0,255,nothing)
cv2.createTrackbar('Red Maximo','Parametros1',0,255,nothing)
 
#Recordamos al usuario con qué tecla se sale:
print("\nPulsa 'ESC' para salir\n")
 
 
while(1):
  #Leemos los sliders y guardamos los valores de H,S,V para construir los rangos:
  BMin1 = cv2.getTrackbarPos('Blue Minimo','Parametros1')
  BMax1 = cv2.getTrackbarPos('Blue Maximo','Parametros1')
  GMin1 = cv2.getTrackbarPos('Green Minimo','Parametros1')
  GMax1 = cv2.getTrackbarPos('Green Maximo','Parametros1')
  RMin1 = cv2.getTrackbarPos('Red Minimo','Parametros1')
  RMax1 = cv2.getTrackbarPos('Red Maximo','Parametros1')
 
  #Creamos los arrays que definen el rango de colores:
  color_bajos1=np.array([BMin1,GMin1,RMin1])
  color_altos1=np.array([BMax1,GMax1,RMax1])
 
  #Detectamos los colores y eliminamos el ruido:
  mask = cv2.inRange(img, color_bajos1, color_altos1)
  mask_1 = cv2.cvtColor(np.uint8(mask), cv2.COLOR_GRAY2RGB)
  mask_1 = cv2.bitwise_or(mask_1, img)
  
  #Mostramos los resultados y salimos:
  cv2.imshow('Original',img)
  cv2.imshow('Mascara1',mask)
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
    break
cv2.destroyAllWindows()