# -*- coding: utf-8 -*-
'''
in tensor flow you have the following
1-iput
2-weight
3-hidden layer 1
4-hidden layer2
comparing output Layers output to the intended output> cost function( crosss entropy)
optimization function(optimizer) > minimizer cost(AdamOptimizer,...SGD, AdaGrad)

backpropagation- optimizers go backwards and manipulate the weights 
feed forward+backprop=epoch
latst version installation, pip install --upgrade tensorflow
'''
import tensorflow as tf
import tensorflow_datasets
#for importing mnist in tensor 2.00 use the following way
mnist=tf.keras.datasets.mnist #28*28 resolution images of handwritten digits 0-9
#we need to unpack that data
(x_train,y_train),  (x_test,y_test) =  mnist.load_data()
print(x_train)
#we neeed to scale or normalize our data for machine to better learn 
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
#build the model, sequential model
model=tf.keras.models.Sequential()
#we want our data to be flat, than multidimensional
#input layer, now we add our models
model.add(tf.keras.layers.Flatten())
#hidden layer, neurons, here we go with two hidden layers,128 neurons or units added,acitvation is the thing whihc makes the neuron fire
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#output layer, it will have the number of classifications
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#need to define some paramters for training of the model, loss is degree of error
#metrics is what we want to track,a NN does not try to maximize accuracy but it try to minimize the loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#now we can train the model
model.fit(x_train,y_train,epochs=3)
#now to calculate loss,accuracy, the accuracy should not be too close nor too far
val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_acc,val_loss)
#to save a model
model.save('epic_num_model.model')
#to reload
new_model=tf.keras.models.load_model('epic_num_model.model')
#to predict, rememeber predict always gets a list
predictions=new_model.predict([x_test])
#to see the prediction result, the print(predictions) shows a machine result we need to make it readable by the numpy calc
import numpy as np
print(np.argmax(predictions[0]))

import matplotlib.pyplot as plt


plt.imshow(x_test[0])
plt.show()
#above should show the 7


#to plot it for better understanding,cmap gives us a black and whites color
plt.imshow (x_train[0],cmap=plt.cm.binary)
plt.show()


