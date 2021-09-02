import cv2
import numpy as np


#Variables
cont_fires=0
cont_nofires=0
cont_aux=0
area=0
cont_img=0

#Ciclo que permite pasar todas las imágenes por el clasificador
for i in range(1,2651):

    #Cargamos la imagen:
    img = cv2.imread("/media/Datos/Escritorio/Documentos tesis/Documento_final_tesis/base de datos termicas/base_datos_termicas_final/incendios/Ter" + str(i) + ".jpg")
    print(i)
    #Definimos rango de máscaras
    negro_morado_bajos = np.array([0,0,0])
    negro_morado_altos = np.array([146,16,157])
    naranja_amarillo_bajos = np.array([0,0,102])
    naranja_amarillo_altos = np.array([95, 238, 255])

    #Detectamos los píxeles que estén dentro del rango que hemos establecido:
    mask1 = cv2.inRange(img, naranja_amarillo_bajos, naranja_amarillo_altos)
    mask2 = cv2.inRange(img, negro_morado_bajos, negro_morado_altos)
    
    #Mascaras para varios colores 
    mask = cv2.bitwise_or(mask1, mask2)

    # obtener los contornos
    contours,ww = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Dibujar contorno rectangular alrededor del incendio
    for c in contours:
        area = cv2.contourArea(c)

        if area > 0.4 and area < 21:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 1, cv2.LINE_AA)
            if cont_img<i:
                cont_fires+=1
                cont_img=i
            area=0
            #break
        if area > 20 and area < 100000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x-5, y-5), (x+5 + w, y+5 + h), (0, 255, 0), 1, cv2.LINE_AA)
            if cont_img<i:
                cont_fires+=1
                cont_img=i
            area=0
    cv2.imwrite("/media/Datos/Escritorio/Documentos tesis/Documento_final_tesis/base de datos termicas/imagenes_detectadas/Ter" + str(i) + ".jpg", img)               
    if cont_aux<cont_fires:
        cont_aux=cont_fires
    else:
        cv2.imwrite("/media/Datos/Escritorio/Documentos tesis/Documento_final_tesis/base de datos termicas/imagenes_nodetectadas/Ter" + str(i) + ".jpg", img)
        cont_nofires+=1

print('Detectados ',cont_fires)
print('No Detectados ',cont_nofires)
