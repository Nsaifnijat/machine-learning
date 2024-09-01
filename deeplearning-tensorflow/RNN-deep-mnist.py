# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Sequential
#dropout is for equalling weighting 
from tensorflow.keras.layers import Dense, Dropout, LSTM #,CuDNNLSTM #on GPU you can use CUDNN instead of LSTM which is faster
from keras.layers import Input
mnist=tf.keras.datasets.mnist
(x_train,y_train), (x_test, y_test) =mnist.load_data()

#we need to normalize our data
x_train=x_train/255.0
x_test=x_test/255.0


'''
#to see how our data is oreder
print(x_train.shape)
print(x_train[0].shape)
'''
#create our model

model=Sequential()
#WE GONNA ADD LSTM LAYER, for GPU CUDNNLSTM is also available and more optimized and fast
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))

#The following is for CuDNNLSTM
#model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))

#to weight
model.add(Dropout(0.2))
#another layer
model.add(LSTM(128, activation='relu'))
#for CuDNNLSTM
#model.add(CuDNNLSTM(128, activation='relu'))

model.add(Dropout(0.2))

#to add a dense layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
#final dense layer, nodes is 10 it is based on how many classes you have
model.add(Dense(10, activation='softmax'))
#need to do the compile now, first need to optimize, lr is learning rate, decay is the rate decrease overe time
opt=tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.fit(x_train,y_train, epochs=3, validation_data=(x_test,y_test))












