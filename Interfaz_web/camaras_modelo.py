from picamera import PiCamera
from base_camera import BaseCamera
from base_camera2 import BaseCamera2

from picamera.array import PiRGBArray

from uvctypes import *
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import cv2
import numpy as np

from torch.autograd import Variable

from time import time

#from sensor_temperatura import ambiente

########
from multiprocessing import Pool
from time import sleep
from dhtxx import DHT11



########


class Camera1(BaseCamera):
 
    @staticmethod
    def frames():
        start_time = time()

        camera = PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 15
        rawCapture = PiRGBArray(camera, size=(320, 240))

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            
            img = frame.array
            
            tiempo_transcurrido = time() - start_time
            
            if tiempo_transcurrido > 8 and tiempo_transcurrido < 8.09:
                print("TOMA_IMAGEN1")
                print('img' + str(tiempo_transcurrido))
                img=cv2.resize(img, (640, 480))
                cv2.imwrite('/home/pi/Desktop/Interfaz_web/static/assets/img/img_prueba.jpg', img)    
                img=cv2.resize(img, (320, 240))
                
            if tiempo_transcurrido >= 10:
                print('img' + str(tiempo_transcurrido))
                start_time = time()      
            
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)
            
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield frame

class Camera2(BaseCamera2):
    
    @staticmethod
    def frames():
        
        BUF_SIZE = 2
        q = Queue(BUF_SIZE)
        
        dht11 = DHT11(17)
        temp_amb=0

        def py_frame_callback(frame, userptr):

            array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
            data = np.frombuffer(
            array_pointer.contents, dtype=np.dtype(np.uint16)
            ).reshape(
            frame.contents.height, frame.contents.width
            ) 

            if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
                return

            if not q.full():
                q.put(data)

        PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

        def generate_colour_map():

            #Se dividio manualmente cada canal de color en el mapa de color ironblack

            lut = np.zeros((256, 1, 3), dtype=np.uint8)

            lut[:, 0, 0] = [255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,143,142,141,140,139,138,137,136,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5,2,0,100,103,106,109,112,115,118,121,124,127,130,133,136,139,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            lut[:, 0, 1] = [255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,143,142,141,140,139,138,137,136,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236]

            lut[:, 0, 2] = [255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,143,142,141,140,139,138,137,136,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5,2,0,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]


            return lut

        def raw_to_8bit(data):

            data = (data/64).astype("uint8")            #Aqui se pasa la matriz de 14 bits a 8 bits

            minVal2, maxVal2, minLoc, maxLoc = cv2.minMaxLoc(data)    #Pixel mayor y menor detectado (para pruebas) 

            return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)
        
                
        def display_temperature(img1, val_raw, amb_aux, loc, color):
            val= 0.0220*val_raw + 1.9403*amb_aux-184.3843
            x, y = loc
            if x<=250 and y<=180:
                loc=(x+20,y+20)
            if x>250 and y<=180:
                loc=(x-60,y+20)
            if x<=250 and y>180:
                loc=(x+20,y-20)
            if x>250 and y>180:
                loc=(x-60,y-20)
            if val>200:
                cv2.putText(img1,">200 C", loc, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            else:
                cv2.putText(img1,"{0:.1f} C".format(val), loc, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            
            
            #x, y = loc
            #cv2.line(img, (x - 10, y), (x + 10, y), color, 1)
        
        
        ###################################################################
        def ambiente (x):
            global temp_amb

            temp_hum = dht11.get_result(max_tries=5)  # 'max_tries' defaults to 5

            if temp_hum:
                print('Temp: {0:0.1f} C'.format(temp_hum[0]))
                temp_amb=temp_hum[0]
                print("AMB_OBTENIDA= ",temp_amb)
                return temp_amb
            else:
                print("AMB_AUXILIAR= ",temp_amb)
                return temp_amb
        
        
        ###################################################################
        
        
        ctx = POINTER(uvc_context)()
        dev = POINTER(uvc_device)()
        devh = POINTER(uvc_device_handle)()
        ctrl = uvc_stream_ctrl()

        res = libuvc.uvc_init(byref(ctx), 0)
        if res < 0:
          print("uvc_init error")
          exit(1)

        try:
          res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
          if res < 0:
            print("uvc_find_device error")
            exit(1)

          try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
              print("uvc_open error")
              exit(1)

            print("device opened!")

            print_device_info(devh)
            print_device_formats(devh)

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
              print("device does not support Y16")
              exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
              frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
            )

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
              print("uvc_start_streaming failed: {0}".format(res))
              exit(1)
            
            try:
              start_time = time()
              while True:
                data = q.get(True, 250)
                if data is None:
                  break
                data = cv2.resize(data[:,:], (320, 240))
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                img = cv2.LUT(raw_to_8bit(data), generate_colour_map())
                amb_aux=20
                display_temperature(img, maxVal, amb_aux, maxLoc, (0, 255, 0))
                ############### mascaras##############3      
                
                tiempo_transcurrido = time() - start_time

                if tiempo_transcurrido > 8 and tiempo_transcurrido < 8.05:  
                   print("TOMA_IMAGEN2")
                   
                   ######
                   if __name__ == '__main__':
                       with Pool(1) as p:
                           probe=p.map(ambiente, [1])
                           amb_aux=probe[0]
                   ######
                   
                   #amb_aux=ambiente()
                   print('ter' + str(tiempo_transcurrido))
                   img1=img
                   #display_temperature(img1, maxVal, amb_aux, maxLoc, (255, 0, 0))
                   img1=cv2.resize(img1, (640, 480))
                   
                   cv2.imwrite('/home/pi/Desktop/Interfaz_web/static/assets/img/ter_prueba.jpg', img1)    
                   img1=cv2.resize(img1, (320, 240))
                   
                if tiempo_transcurrido >= 10:
                   print('ter' + str(tiempo_transcurrido))
                   start_time = time()

                frame = cv2.imencode('.jpg', img)[1].tobytes()
                yield frame

              cv2.destroyAllWindows()
            finally:
              libuvc.uvc_stop_streaming(devh)

            print("done")
          finally:
            libuvc.uvc_unref_device(dev)
        finally:
          libuvc.uvc_exit(ctx)
