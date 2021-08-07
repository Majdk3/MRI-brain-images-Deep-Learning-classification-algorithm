
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy
import os
import cv2

import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

def predct(image_dirc):
    model = load_model('C:/Users/ASUS/Desktop/test application/models/ensample.h5')


    img=cv2.imread(image_dirc,1)
    
    img=cv2.resize(img,(512,512))
    img=img/255.0
    img=numpy.reshape(img,(-1,512,512,3))
    x=0
    prediction_value=model.predict(img)
    prediction_value=np.argmax(prediction_value,axis=1)
    if prediction_value==0:
        x=1
        return "Meningiomas",x
    elif prediction_value==1:
        x=2
        return "Gliomas",x
    elif prediction_value==2:
        x=3
        return "Pituitary tumors",x
    
