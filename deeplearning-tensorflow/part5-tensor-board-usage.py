# -*- coding: utf-8 -*-
#if you want to specify a certain GPU size for any of your algos, do the first two lines

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess=tf.session(config=tf.ConfigProto(gpu_options=gpu_options))

NAME='Cats-vs-dog-cnn-64x2-{}'.format(int(time.time()))

tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

X=pickle.load(open('X.pickle','rb'))
y=pickle.load(open('y.pickle','rb'))
#first you need to normalizer your data
X=X/255.0

model=Sequential()
#convolutional layer=conv2D, 64 is units,(3,3) is window size, input_shape dynamically get input shape
model.add(  Conv2D(64,(3,3),input_shape=X.shape[1:])  )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#second layer, now we have a 2x64 convolutionalNN 
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#conv2D is 2D, now we need to flatten our data
model.add(Flatten())#this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

#output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#batch size is in how many batches do you want to pass the data to the machine
model.fit(X,y,batch_size=32,epochs=10,validation_split=0.1,callbacks=[tensorboard])

#to run the tensorboard log dir
#tensorboard --logdir=logs/











































