import cv2
import os
import numpy as np
import time

def modelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer	 = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+".xml")

dataPath = 'C:/Users/din_p/OneDrive/Escritorio/ResultadosEmociones' #Cambia a la ruta donde hayas almacenado Data
listaEmociones = os.listdir(dataPath)
print('Lista de emociones: ', listaEmociones)

labels = []
facesData = []
label = 0

for nameDir in listaEmociones:
	emotionsPath = dataPath + '/' + nameDir
	for fileName in os.listdir(emotionsPath):
		labels.append(label)
		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
		#image = cv2.imread(emotionsPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

modelo('EigenFaces',facesData,labels)
modelo('FisherFaces',facesData,labels)
modelo('LBPH',facesData,labels)