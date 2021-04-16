# -*- coding: utf-8 -*-
"""PartA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gudKWoYRTiyMKBuQUpj6rVLAcHRG-Zhb
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

!pip install wandb

import wandb

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_a = ImageDataGenerator(rescale=1./255,validation_split=0.1, rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest').flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=5000,
                                                     class_mode='binary',
                                                     subset='training',
                                                     seed=12)

val_a = ImageDataGenerator(rescale=1./255,validation_split=0.1, rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest').flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=200,
                                                     class_mode='binary',
                                                     subset='validation',
                                                     seed=12)

X_at,y_at=next(train_a)

x_av,y_av=next(val_a)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_generator = ImageDataGenerator(rescale=1./255,validation_split=0.1).flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=5000,
                                                     class_mode='binary',
                                                     subset='training',
                                                     seed=12)
val_generator = ImageDataGenerator(rescale=1./255,validation_split=0.1).flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=200,
                                                     class_mode='binary',
                                                     subset='validation',
                                                     seed=12)

X_train, y_train = next(train_generator)

X_val, y_val = next(val_generator)

train_test = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest').flow_from_directory('/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/val',target_size=(224, 224),batch_size=1000,class_mode='binary',seed=12)

x_test,t_test=next(train_test)

class MyCNN(object):
  def __init__(self,Dropout,Layer1,Layer2,Layer3,Layer4,Layer5,bn,augmentation):
     self.Dropout=Dropout
     self.Layer1=Layer1
     self.Layer2=Layer2
     self.Layer3=Layer3
     self.Layer4=Layer4
     self.Layer5=Layer5
     self.bn=bn
     self.augmentation=augmentation
  def train(self):
    input_shape=(224,224,3)
   
    Layer=[self.Layer1,self.Layer2,self.Layer3,self.Layer4,self.Layer5]
    
    model =models.Sequential()
    model.add(layers.Conv2D(Layer[0], (3, 3), input_shape=input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[1], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[2], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[3], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[4], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(self.Dropout))

    model.add(layers.Dense(10,activation='softmax'))  
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
  
    if(self.augmentation=='Yes'):
      model.fit(X_at,y_at, epochs=15,validation_data=(x_av, y_av),callbacks=[WandbCallback(validation_data=(x_av, y_av))])
    if(self.augmentation=='No'):
      model.fit(X_train,y_train, epochs=15,validation_data=(X_val, y_val),callbacks=[WandbCallback(validation_data=(X_val, y_val))])

from wandb.keras import WandbCallback

wandb.init()

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'accuracy',
        
    },
    'parameters': {
        'Dropout': {
            'values': [0.2,0.3]
        },
        
        
        
        'Layer1': {
            'values': [16,32,64,128,256]
        },

        'Layer2': {
            'values': [16,32,64,128,256]
        },

        
        'Layer3': {
            'values': [16,32,64,128,256]
        },

        
        'Layer4': {
            'values': [16,32,64,128,256]
        },

        
        'Layer5': {
            'values': [16,32,64,128,256]
        },

        'Batch_Normalisation': {
            'values': ['Yes','No']
        },
        'Augmentation': {
            'values': ['Yes','No']
        },
    }
}


sweep_id = wandb.sweep(sweep_config, entity='hemprakashpatidar', project="My Assign2")

wandb.init()

wandb.sweep(sweep_config)
'''config_defaults = {
            'epochs': 5,
            'learning_rate': 1e-2
            
          }
'''

def train_m():
  run = wandb.init()
  config=run.config
  modell = MyCNN(config.Dropout,config.Layer1,config.Layer2,config.Layer3,config.Layer4,config.Layer5,config.Batch_Normalisation,config.Augmentation)
          
  modell.train()
    
  

wandb.agent(sweep_id,train_m)

input_shape=(224,224,3)
Layer=[16,256,32,16,128]
model =models.Sequential()
model.add(layers.Conv2D(Layer[0], (3, 3), input_shape=input_shape))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())


model.add(layers.Conv2D(Layer[1], (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())


model.add(layers.Conv2D(Layer[2], (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())


model.add(layers.Conv2D(Layer[3], (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())


model.add(layers.Conv2D(Layer[4], (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(10,activation='softmax'))

model.evaluate(x_test,t_test)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20,
                    
                    validation_data=(X_val, y_val))

labels=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

test_img=[]
predict_label=[]
for i in range(0,30):
  test_image = x_test[i]
  test_image = np.reshape(test_image,(1,224,224,3))
  result = model.predict(test_image)
  test_img.append(x_test[i].reshape(224,224,3))
  if(np.argmax(result)==t_test[i]):
    predict_label.append(labels[np.argmax(result)]+'\t'+'True')
  else:
    predict_label.append(labels[np.argmax(result)]+'\t'+'False')

wandb.init()

wandb.log({"Prediction": [wandb.Image(img,caption=caption) for img,caption in zip(test_img,predict_label)]})

model.evaluate(x_test,t_test)