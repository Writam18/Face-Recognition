from keras.models import model_from_json
import cv2
from model import create_model
from Align import AlignDlib
import numpy as np

'''
import dlib
import pickle
from PIL import Image
import base64
import io
import keras
import tensorflow as tf
import os
import pandas as pd
'''
alignment = AlignDlib('landmarks.dat')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
names = np.load('names.npy')
le.fit(names)

def get_model():
    global model_clf
    json_file = open('W_FYP_FR_model_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_clf = model_from_json(loaded_model_json)
    model_clf.load_weights("W_FYP_FR_weights_1.h5")
    model_clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Loaded NN model from disk")

def get_openface_model():
	global model_of
	model_of = create_model()
	model_of.load_weights('open_face.h5')
	#global graph
	#graph = tf.get_default_graph()
	print('Loaded openface model')

def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]
    
def predict_face(path):
  img = load_image(path)
  faces = alignment.getAllFaceBoundingBoxes(img)
  print(faces)
  for i in range(len(faces)):
    face_aligned = alignment.align(96, img, faces[i], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    face_aligned = (face_aligned / 255.).astype(np.float32)
    embedding = model_of.predict(np.expand_dims(face_aligned, axis=0))
    pred = model_clf.predict([[embedding]])
    ind = np.argsort(pred[0])
    print(ind[::-1][:5])
    if(pred[0][ind[::-1][0]]*100>70):
      print("Prediction: ",le.inverse_transform([ind[::-1][0]])[0])
      print("Prediction Probability: ",pred[0][ind[::-1][0]]*100,"%")
      print()
    else:
      print("Prediction: Others")
      print("Prediction Probability: ",pred[0][ind[::-1][0]]*100,"%")
      print()


get_model()
get_openface_model()

path3 = '/home/writam/FYP-FR/Photos2_test/PJ1.jpg'
predict_face(path3)

