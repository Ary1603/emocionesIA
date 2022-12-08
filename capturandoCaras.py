import cv2
import numpy as np
import imutils
import os

personaNombre = 'GerardoEnojado'
dataPath = 'C:/Users/din_p/OneDrive/Escritorio/Resultados/AdrianTriste'
personaRuta = dataPath 
#Si no existe el archivo
if not os.path.exists(personaRuta):
    print('Carpeta creada: ', personaRuta)
    os.makedirs(personaRuta)


captura = cv2.VideoCapture("C:/Users/din_p/OneDrive/Escritorio/Resultados/AdrianTriste/AdrianTriste.mp4")
clasificacion = cv2.CascadeClassifier('index.xml')

count = 0

while True:
    ret, frame = captura.read()
    if ret == False: break
    frame = imutils.resize(frame,width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    
    faces = clasificacion.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personaRuta + '/rostroA_{}.jpg'.format(count),rostro)
        count = count + 1
        
    #Abrir ventana
    cv2.imshow('NOMA IA',frame)
    
    #Establecer el número de imagenes a crear
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
    
    
captura.release()
cv2.destroyAllWindows()
    

