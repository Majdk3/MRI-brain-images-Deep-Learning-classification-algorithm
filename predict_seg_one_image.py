

import cv2
import tensorflow as tf
import numpy as np
from Unet import Unet, dice_score
#from generator import DataGenerator
import os
def predict_seegmented(img_dirc):
    print(img_dirc)
    if not os.path.isfile(img_dirc):
        
        return print("file not  found")

    #test_data = DataGenerator('C:/Users/ASUS/Desktop/New folder/project 2.0/data set/matlab files/2.mat')

    print("file found")

    model = Unet((512, 512, 1), n_filters=16, dropout=None).model
    img=cv2.imread(img_dirc)
    img=cv2.resize(img,(512,512))
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img=np.reshape(img,(-1,512,512,1))


    model.load_weights(
            'C:/Users/ASUS/Desktop/test application/models/model_unet.h5')

    predictions = model.predict(
            img, steps=1, verbose=1)
    predictions=predictions[0,:,:]
    return predictions
    
