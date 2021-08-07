

import keras
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf 
from keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from keras.regularizers import l2

#this  function will create a model arch same as the one used to build the orginal models
#basically you will have n functions related to n numbers of models used
#input line and output and model load weights must be there always



#important note: in this case the models used MUST be trained on the same data cofiguration so no model will be trained on the test data
def get_model():
    #load first model
    input1=keras.Input(shape=(512,512,3))

    c1=Conv2D(8,(3,3))(input1)
    d1=Dropout(0.1)(c1)
    a1=LeakyReLU(alpha=0.1)(d1)
    b1=BatchNormalization()(a1)
    p1=MaxPool2D(pool_size=(1,2))(b1)
    
    c2=Conv2D(16,(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p1)
    d2=Dropout(0.2)(c2)
    a2=LeakyReLU(alpha=0.1)(d2)
    b2=BatchNormalization()(a2)
    p2=MaxPool2D(pool_size=(3,2))(b2)
    
    
    c3=Conv2D(32,(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p2)
    d3=Dropout(0.3)(c3)
    a3=LeakyReLU(alpha=0.1)(d3)
    b3=BatchNormalization()(a3)
    p3=MaxPool2D(pool_size=(2,2))(b3)

    f1=Flatten()(p3)
    D1=Dense(32,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(f1)
    a4=LeakyReLU(alpha=0.1)(D1)
    b4=BatchNormalization()(a4)
    
    
    d4=Dropout(0.3)(b4)
    output=Dense(3,activation='softmax')(d4)

    
    model= keras.Model(inputs=input1, outputs=output)
    
    model.load_weights('C:\\Data set 4\\checkpoint\\best_val_accurcyb10_e100_3.h5')
    return model




def get_model1():
    #load second model
    input1=keras.Input(shape=(512,512,3))

    c1=Conv2D(8,(3,3))(input1)
    d1=Dropout(0.1)(c1)
    a1=LeakyReLU(alpha=0.1)(d1)
    b1=BatchNormalization()(a1)
    p1=MaxPool2D(pool_size=(1,2))(b1)
    
    c2=Conv2D(16,(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p1)
    d2=Dropout(0.2)(c2)
    a2=LeakyReLU(alpha=0.1)(d2)
    b2=BatchNormalization()(a2)
    p2=MaxPool2D(pool_size=(3,2))(b2)
    
    
    c3=Conv2D(32,(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p2)
    d3=Dropout(0.3)(c3)
    a3=LeakyReLU(alpha=0.1)(d3)
    b3=BatchNormalization()(a3)
    p3=MaxPool2D(pool_size=(2,2))(b3)

    f1=Flatten()(p3)
    D1=Dense(32,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(f1)
    a4=LeakyReLU(alpha=0.1)(D1)
    b4=BatchNormalization()(a4)
    
    
    d4=Dropout(0.3)(b4)
    output=Dense(3,activation='softmax')(d4)

    
    model= keras.Model(inputs=input1, outputs=output)
    
    model.load_weights('C:\\Data set 4\\checkpoint\\best_val_accurcyb10_e100_1.h5')
    return model

def get_model2():
    #load third model
    input1=keras.Input(shape=(512,512,3))

    c1=Conv2D(8,(3,3))(input1)
    d1=Dropout(0.1)(c1)
    a1 = LeakyReLU(alpha=0.1)(d1)
    b1=BatchNormalization()(a1)
    p1=MaxPool2D(pool_size=(1,2))(b1)
    
    c2=Conv2D(16,(3,3),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p1)
    d2=Dropout(0.2)(c2)
    a2 = LeakyReLU(alpha=0.1)(d2)
    b2=BatchNormalization()(a2)
    p2=MaxPool2D(pool_size=(3,2))(b2)
    
    
    c3=Conv2D(32,(3,3),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(p2)
    d3=Dropout(0.3)(c3)
    b3=BatchNormalization()(d3)
    p3=MaxPool2D(pool_size=(2,2))(b3)

    f1=Flatten()(p3)
    D1=Dense(32,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(f1)
    b4=BatchNormalization()(D1)
    
    
    d4=Dropout(0.3)(b4)
    output=Dense(3,activation='softmax')(d4)

    
    model= keras.Model(inputs=input1, outputs=output)
    
    model.load_weights('C:\\Data set 4\\checkpoint\\best_val_accurcyb10_e100_5.h5')
    return model

#here we try only one image as a test, the result will be as three probilties of the three classes
import cv2
import numpy 
img=cv2.imread('C:\\Data set 4\\train\\1\\30.matlabel=1.png',1)

img=cv2.resize(img,(512,512))
img=img/255.0
img=numpy.reshape(img,(-1,512,512,3))

#creating the ensambling model
inputs = keras.Input(shape=(512,512,3))


model1=get_model()
model2 = get_model1()
model3 = get_model2()

y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = keras.layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

#printing the result of prediticng that single image
pred=ensemble_model.predict(img)
print(pred)









test_data_dir='C:\\Data set 4\\test'
nb_train_samples = 2400
nb_validation_samples = 301

batch_size = 10

#here we create the test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(512, 512),
    shuffle=False,
    batch_size=batch_size,
    class_mode='categorical')
#here we test the whole test data
#printing the test acc,all parameters needed

import numpy as np


from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


ensemble_model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy','categorical_crossentropy','categorical_accuracy'])

score = ensemble_model.evaluate_generator(test_generator, verbose=1)
print("%s: %.2f%%" % (ensemble_model.metrics_names[1], score[1]*100))
Y_pred = ensemble_model.predict_generator(test_generator, 301 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(test_generator.classes, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal(),"class acurcy")
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['mil', 'gilomas', 'pit tumors']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
