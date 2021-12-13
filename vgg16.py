import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# EDA
len(os.listdir("/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid/"))

train_datagen = ImageDataGenerator(zoom_range= 0.5, 
                                   shear_range= 0.2, 
                                   rescale= 1/255,
                                   horizontal_flip= True, 
                                   preprocessing_function= preprocess_input ) #change some values to get desired images, rescale one see

val_datagen = ImageDataGenerator(zoom_range= 0.5, 
                                   shear_range= 0.2, 
                                   rescale= 1/255,
                                   horizontal_flip= True, preprocessing_function= preprocess_input )
train = train_datagen.flow_from_directory(directory="/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid", 
                                          target_size= (256,256), 
                                          batch_size= 8)

val = val_datagen.flow_from_directory(directory= "/content/drive/MyDrive/dataset1/New Plant Diseases Dataset(Augmented)/valid", 
                                      target_size= (256,256), 
                                      batch_size= 8)
t_img, label = train.next()
t_img.shape

def plotImage(img_arr, label):

  for im, l in zip(img_arr , label):
    plt.figure(figsize=(5,5))
    plt.imshow(im)

plotImage(t_img[:3], label[:3])

v_img, label1 = val.next()
v_img.shape

def plotImage(img_arr1, label1):

  for im1, l in zip(img_arr1 , label1):
    plt.figure(figsize=(5,5))
    plt.imshow(im1)

plotImage(t_img[:3], label[:3])

from keras.layers import Dense, Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16
import keras

base_model = VGG16(input_shape=(256,256,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False

base_model.summary()

X = Flatten()(base_model.output)

X = Dense(units= 38, activation='softmax')(X)


#Creating our model
model = Model(base_model.input, X)

model.summary()
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#check

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
#network gets trained here
his = model.fit_generator(train, 
                          steps_per_epoch= 24,  
                          epochs= 50, 
                          verbose=1,
                          callbacks= cb, 
                          validation_data=val,
                          validation_steps= 24)

#plotting metrics
h = his.history
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

model = load_model("/content/drive/MyDrive/dataset1/model_trained/resnet_model/best_model.h5")

#to get accuracy
acc = model.evaluate_generator(val)[1]
print(f"The accuracy of your model is {acc*100}%")

train.class_indices
ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys() ) ))

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
