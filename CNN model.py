from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.utils.vis_utils import plot_model
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from keras.regularizers import l2
from keras.callbacks import Callback
from keras.models import load_model

name_of_model = input("please write the name of the model ")

tensorboard =tf.keras.callbacks.TensorBoard(log_dir='C:\\Data set 4\\new models\\logs\\{}'.format(name_of_model))

# this is so the code can use as much vram as possible
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# dimensions of our images.
img_width, img_height = 512, 512

# enter the location of the data below, here it is seperated to several folders as we used .png files
train_data_dir = 'C:\\Data set 4\\train1'
validation_data_dir = 'C:\\Data set 4\\evaluate1'
test_data_dir = 'C:\\Data set 4\\test1'
# specify the number of training and validation samples and the number of epochs and batch size
nb_train_samples = 2184
nb_validation_samples = 458
epochs = 300
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

# building the CNN model
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=input_shape))
model.add(Dropout(0.1))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1, 2)))

model.add(Conv2D(16, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3, 2)))

model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, kernel_regularizer=regularizers.l2(1e-4), bias_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()
plot_model(model, to_file='C:\\Data set 4\\new models\\model_plot_{}.png'.format(name_of_model), show_shapes=True,
           show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', 'categorical_crossentropy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

csvloger = tf.keras.callbacks.CSVLogger("C:\\Data set 4\\new models\\new models{}.csv".format(name_of_model))
# the function below reduces the learning rate when the catagorical crossentropy plateaus to help the model converge better.
reduceLRO = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.3,
                                                 patience=10, cooldown=10, min_lr=0.00001, verbose=1)



# other callbacks

#checkpoints are used to perserve the model with the best test results to avoid overfitting
checkpoint = ModelCheckpoint('C:\\Data set 4\\checkpoint\\best_val_accurcy{}.h5'.format(name_of_model),
                             monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[csvloger, reduceLRO, checkpoint, tensorboard],
    validation_steps=nb_validation_samples // batch_size)
model.save('C:\\Data set 4\\new models\\{}.h5'.format(name_of_model))




test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# evaluate the model
score = model.evaluate_generator(test_generator, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

