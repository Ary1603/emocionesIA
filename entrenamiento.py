import cv2
import os
import numpy as np
import imutils

dataPath = 'C:/Users/din_p/OneDrive/Escritorio/Resultados' #Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
infoCaras = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		infoCaras.append(cv2.imread(personPath+'/'+fileName,0))
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

#Método LBPHF
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(infoCaras, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('ModeloLBPHFace.xml')
print("Modelo almacenado...")
        