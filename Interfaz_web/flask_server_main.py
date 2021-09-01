#!/usr/bin/env python
#import Adafruit_DHT
from dhtxx import DHT11, DHT22

#Librerias flask
from flask import Flask, render_template, Response
from flask import jsonify

#Libreria flask con hilos
from flask_executor import Executor

#Scripts de camaras
from camaras_modelo import Camera1, Camera2

#Librerias para procesar imagenes
import cv2
import numpy as np
from PIL import Image

#Librerias GPS
import serial
import pynmea2

#Librerias de pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable

#Libreria base de datos
import mariadb


import sys
from datetime import datetime
import shutil
import psutil
import time

app = Flask(__name__)
executor = Executor(app)

app.config['EXECUTOR_TYPE'] = 'process'
app.config['EXECUTOR_MAX_WORKERS'] = 5


cont_aux=0
amb_aux=0

#Conexión base de datos

try:
    mi_base_datos = mariadb.connect(
      user="daniel",
      password="Tesis@2021",
      host="localhost",
      database="Registros_incendios")
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

cursor=mi_base_datos.cursor()
cursor.execute('select * from historial')
resultado=cursor.fetchall()
try:
    contador = resultado[-1][0]
except:
    contador = 0
#Renderizar index
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


tiempo_inicio = time.time()

#Función de alertas
@app.route('/alerta', methods = ['GET'])
def stuff():
    #Variable para no realizar predicción apenas inicie la ejecición del programa
    global cont_aux
    global amb_aux
    global tiempo_inicio
    
    
    tiempo_espera = time.time() - tiempo_inicio
    #Función para obtener coordenadas de GPS
    def coordinates():
        
        while True:         
                        
            lat='0'
            lng='0'
            
            port="/dev/ttyAMA0"
            ser=serial.Serial(port, baudrate=9600, timeout=0.5)
            newdata=ser.readline()
                
            if newdata[0:6] == b'$GPRMC':
                newdata = newdata.decode(encoding = 'utf-8')
                newmsg=pynmea2.parse(newdata)
                lat=str(round(newmsg.latitude,7))
                lng=str(round(newmsg.longitude,7))
                print(lat + " " + lng)
                return(lat, lng)
     
                        
    ubication= executor.submit(coordinates)   
    ubication=ubication.result()
    
    if tiempo_espera>=40:
        def prediction_term():
            print("PREDICCION_TERMICA")
            aux_term=0
            img = cv2.imread("/home/pi/Desktop/Interfaz_web/static/assets/img/ter_prueba.jpg")
            
            #Colores máscaras
            
            #negro_morado_bajos = np.array([0,0,0])
            #negro_morado_altos = np.array([146,16,157])

            #naranja_amarillo_bajos = np.array([0,0,102])
            #naranja_amarillo_altos = np.array([95, 238, 255])
            
#             negro_morado_bajos = np.array([0,0,0])
#             negro_morado_altos = np.array([139,50,195])
# 
#             naranja_amarillo_bajos = np.array([0,50,140])
#             naranja_amarillo_altos = np.array([130, 230, 255])
            
            negro_morado_bajos = np.array([0,0,0])
            negro_morado_altos = np.array([150,76,180])

            naranja_amarillo_bajos = np.array([0,0,140])
            naranja_amarillo_altos = np.array([75, 240, 255])
            
            #Detectamos los píxeles que estén dentro del rango que hemos establecido:
            
            mask1 = cv2.inRange(img, naranja_amarillo_bajos, naranja_amarillo_altos)
            mask2 = cv2.inRange(img, negro_morado_bajos, negro_morado_altos)
         
            #Mascaras para varios colores 
            mask = cv2.bitwise_or(mask1, mask2)

            #Obtener los contornos
            contours,ww = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #Dibujar contorno rectangular alrededor del incendio
            
#             for c in contours:
#                 area = cv2.contourArea(c)
#                 if area > 0.4 and area < 21:
#                     (x, y, w, h) = cv2.boundingRect(c)
#                     cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 1, cv2.LINE_AA)
#                     aux_term=1
# 
#                 elif area > 20 and area < 100000:
#                     (x, y, w, h) = cv2.boundingRect(c)
#                     cv2.rectangle(img, (x-5, y-5), (x+5 + w, y+5 + h), (255, 0, 0), 1, cv2.LINE_AA)
#                     aux_term=1
# 
#                 else:
#                     aux_term=0
            for c in contours:
                area = cv2.contourArea(c)
                if area > 0.4 and area < 100000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    #cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (255, 0, 0), 1, cv2.LINE_AA)
                    aux_term=1
                else:
                    aux_term=0
            if contours==[]:
                aux_term=0
                return 0
            if aux_term==1:
                cv2.imwrite("/home/pi/Desktop/Interfaz_web/static/assets/img/ter_prueba.jpg", img)
                return 1
            else:
                return 0
  
  
        def prediction():
            print("PREDICCIÓN COLOR")
            
            def load_checkpoint(filepath, model_full):
                checkpoint = torch.load(filepath)
                model = torch.load(model_full)
                
                #Modelo Squeezenet
                num_classes=2
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                model.num_classes = num_classes
                
                #Modelo Resnet
                #num_ftrs = model.fc.in_features  #numero de neuronas en la capa de entrada 
                #model.fc = nn.Linear(num_ftrs, 2) #selecciona el numero de clases 
                
                
                model.load_state_dict(checkpoint['state_dict'])
                
                return model, checkpoint['class_to_idx']


            def process_image(image):

                size = 256, 256
                image.thumbnail(size, Image.ANTIALIAS)
                image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
                npImage = np.array(image)
                npImage = npImage/255.
                    
                imgA = npImage[:,:,0]
                imgB = npImage[:,:,1]
                imgC = npImage[:,:,2]
                
                imgA = (imgA - 0.485)/(0.229) 
                imgB = (imgB - 0.456)/(0.224)
                imgC = (imgC - 0.406)/(0.225)
                    
                npImage[:,:,0] = imgA
                npImage[:,:,1] = imgB
                npImage[:,:,2] = imgC
                
                npImage = np.transpose(npImage, (2,0,1))
                return npImage

            def predict(image_path, model, topk=2):

                image = torch.FloatTensor([process_image(Image.open(image_path))])
                model.eval()
                output = model.forward(Variable(image))
                pobabilities = torch.exp(output).data.numpy()[0]
                top_idx = np.argsort(pobabilities)[-topk:][::-1] 
                top_class = [idx_to_class[x] for x in top_idx]
                top_probability = pobabilities[top_idx]
                return top_probability, top_class
            
            
            #Modelo Squeezenet 1_0
            filepath='/home/pi/Desktop/Interfaz_web/model_squeezenet_train.pth'
            model_full='/home/pi/Desktop/Interfaz_web/modelo_squeezenet_full.pth'
            
            #Modelo Resnet18
            #filepath='/home/pi/Downloads/RESNET18 model_It1.pth'
            #model_full='/home/pi/Downloads/modelo_Resnet18_full.pth'
            
            loaded_model, class_to_idx = load_checkpoint(filepath, model_full)
            idx_to_class = { v : k for k,v in class_to_idx.items()}
            
            top_probability, top_class=predict('/home/pi/Desktop/Interfaz_web/static/assets/img/img_prueba.jpg', loaded_model)
            top=top_class[0]
            dif=top_probability[0]-top_probability[1]
            print(top_class)
            print(top_probability)

            if dif > 70000:
                return 1
            else:
                return 0
           
        alert1 = executor.submit(prediction)
        alert2 = executor.submit(prediction_term)
        print("ALERTA 1 ",alert1.result())
        print("ALERTA 2 ",alert2.result())
    
        #Fecha actual
        now = datetime.now()

        day=now.day
        if day < 10:
            now_day="0"+str(day)
        else:
            now_day=str(day)
          
        month=now.month
        if month < 10:
            now_month="0"+str(month)
        else:
            now_month=str(month)
         
        hour=now.hour
        if hour < 10:
            now_hour="0"+str(hour)
        else:
            now_hour=str(hour)
            
        minute=now.minute
        if minute < 10:
            now_minute="0"+str(minute)
        else:
            now_minute=str(minute)

        second=now.second
        if second < 10:
            now_second="0"+str(second)     
        else:
            now_second=str(second)
            
        date=now_day + "-" + now_month + "-" + str(now.year) + " " + now_hour + ":" + now_minute + ":" + now_second
        date_dir=now_day + "-" + now_month + "-" + str(now.year) + "_" + now_hour + ":" + now_minute + ":" + now_second
        
        global contador
         
        if alert1.result() == 1 or alert2.result() == 1:
            mensaje="incendio"
            contador+=1
            print("###########GUARDADO################")
            ruta_color="static/assets/img/img_prueba.jpg"
            ruta_termica="static/assets/img/ter_prueba.jpg"
            
            destino1 = "static/assets/img/save_img/Color-"+ str(date_dir) +".jpg"
            destino2 = "static/assets/img/save_img/Termica-"+ str(date_dir) +".jpg"
            
            shutil.copyfile(ruta_color, destino1)
            shutil.copyfile(ruta_termica, destino2)
            
            sql="INSERT INTO historial(Imagen_color, Imagen_termica, Latitud, Longitud, Fecha) VALUES(?,?,?,?,?)"
            values=(destino1, destino2, ubication[0],ubication[1], date)
            cursor.execute(sql, values)
            mi_base_datos.commit()
            
            return jsonify(contador=contador, mensaje=mensaje, ruta_color=destino1, latitud=ubication[0], longitud=ubication[1], resultado=resultado, date=date, ruta_termica=destino2)
        else:
            mensaje="No incendio"
            return jsonify(contador=' ', mensaje=mensaje, ruta_color=' ', latitud=ubication[0], longitud=ubication[1], resultado=resultado, date=date, ruta_termica=' ')
    else:   
        return jsonify(cont_aux=cont_aux, latitud=ubication[0], longitud=ubication[1], resultado=resultado)


def gen1(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def gen2(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed1')
def video_feed1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen1(Camera1()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen2(Camera2()),mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == '__main__':
    #app.run(host='192.168.137.191', debug=True, threaded=True)
    #app.run(host='192.168.0.13', port=7008, debug=True)
    app.run(host='192.168.1.103', port=7000, debug=True)
    #app.run(host='192.168.254.2', port=7014, debug=True)