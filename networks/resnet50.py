from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline 

import cv2

import os

import numpy as np
from keras.preprocessing import image

NUM_CLASSES = 38

RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']


model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))


# model.add(Dense(1024, activation = DENSE_LAYER_ACTIVATION))
# 2nd layer as Dense for 8-class classification
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

#compiling
model.compile(optimizer = 'adam', loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range = 270,
    brightness_range = (-2,2),
        shear_range=0.2,
        zoom_range=0.3,
    vertical_flip=True,
    validation_split=0.3,
        horizontal_flip=True )

test_datagen = ImageDataGenerator(vertical_flip=True,
                                 horizontal_flip=True,
                                 rotation_range=180,
                                 zoom_range=0.2 )

val_datagen = ImageDataGenerator(vertical_flip=True,
                                 horizontal_flip=True,
                                 rotation_range=180,
                                 zoom_range=0.2 )

train_set = train_datagen.flow_from_directory(
        '/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid',
        target_size=(256,256), #was 224
        batch_size=64,
        seed=101, 
        shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid',
        target_size=(256, 256),
        batch_size=64,
        shuffle=True,
        seed=101,
        class_mode='categorical')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(zoom_range= 0.5, 
                                   shear_range= 0.2, 
                                   rescale= 1/255,
                                   horizontal_flip= True, 
                                   preprocessing_function= preprocess_input)

test_datagen = ImageDataGenerator(zoom_range= 0.5, 
                                   shear_range= 0.2, 
                                   rescale= 1/255,
                                   horizontal_flip= True, preprocessing_function= preprocess_input ) 

val_datagen = ImageDataGenerator(zoom_range= 0.5, 
                                   shear_range= 0.2, 
                                   rescale= 1/255,
                                   horizontal_flip= True, preprocessing_function= preprocess_input ) 

train_set = train_datagen.flow_from_directory(
        '/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid',
        target_size=(256,256), #was 224
        batch_size=64,
        seed=101, 
        shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid',
        target_size=(256, 256),
        batch_size=64,
        shuffle=True,
        seed=101,
        class_mode='categorical')

#for training images
t_img, label = train_set.next()
t_img.shape

def plotImage(img_arr, label):

  for im, l in zip(img_arr , label):
    plt.figure(figsize=(5,5))
    plt.imshow(im)

plotImage(t_img[:3], label[:3])

#for test images
v_img, label1 = test_set.next()
v_img.shape

def plotImage(img_arr, label):

  for im, l in zip(img_arr , label):
    plt.figure(figsize=(5,5))
    plt.imshow(im)

plotImage(v_img[:3], label1[:3])

from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
es = EarlyStopping(monitor= 'val_accuracy', min_delta= 0.001, patience= 10, verbose=1)

# model check point
mc = ModelCheckpoint(filepath="best_model.h5",
                     monitor= 'val_accuracy', 
                     min_delta= 0.001, 
                     patience=10, 
                     verbose=1,
                     save_best_only= True)

cb = [es, mc]

history = model.fit_generator(train_set, 
                          steps_per_epoch= 24, 
                          epochs= 50, 
                          verbose=1, 
                          callbacks= cb, 
                          validation_data=test_set,
                          validation_steps= 24) 

#plotting metrics
h = history.history
h.keys()
    
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c= 'red')
plt.title("acc vs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c= "red")
plt.title("t_loss vs v-loss")
plt.show()

# load best model

from keras.models import  load_model
model = load_model("/content/best_model.h5")

#to get accuracy
acc = model.evaluate_generator(val)[1]
print(f"The accuracy of your model is {acc*100}%")

train_set.class_indices
ref = dict(zip(list(train_set.class_indices.values()), list(train_set.class_indices.keys() ) ))

# for training data
def prediction(path):

  img = load_img(path, target_size= (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)

  img = np.expand_dims(im, axis = 0)

  pred = np.argmax(model.predict(img))
  print(pred)

# for test data
def prediction(path):

  img = load_img(path, target_size= (256,256))

  i = img_to_array(img)

  im = preprocess_input(i)

  img = np.expand_dims(im, axis = 0)

  pred = np.argmax(model.predict(img))
  print(f'the image belongs to {ref[pred]}')

#predicting the image class
path = "/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid/Blueberry___healthy/test.JPG"
prediction(path)


