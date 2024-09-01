# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf

CATEGORIES=['Dog','Cat']

#this func take the img we want to predict and resize and reshape  it then save it into an array
def prepare(filepath):
    IMG_SIZE=50
    #onvert to grayscale
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
#here load your trained model
model=tf.keras.models.load_model('64x3-CNN.model')
#predict your new image, predict paramter has to be a list
prediction=model.predict([prepare('cat.jpg')])

print(CATEGORIES[int(prediction[0][0])])


prediction=model.predict([prepare('dog.jfif')])

print(CATEGORIES[int(prediction[0][0])])