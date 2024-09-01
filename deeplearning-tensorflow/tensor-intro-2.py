# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2#image manipulation

DATADIR='C:\\Users\\Hamdard PC\\allDATA\\Machine Learning\\DataDC\\PetImages'
CATEGORIES=['Dog','Cat']

for category in CATEGORIES:
    path=os.path.join(DATADIR, category) #path to cats or dogs dir
    print(os.listdir(path))
    #LOOP through the images of the dir
    for img in os.listdir(path):
        #read them into an array, make the color grayscale
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        print(img_array)
        #to see what we have
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break
#lets see image array, once in color and once in grayscale
print(img_array)
#to resize the images to 50x50
IMG_SIZE=50
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
    
#create training data
training_data=[]

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category) #path to cats or dogs dir
        class_num=CATEGORIES.index(category)
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

#the training data is better to be balanced between the two category
#its better to shuffle the data to make it better for machine to learn, 
#it makes it not to get stuck with one category and then when the other comes it get confused

import random
random.shuffle(training_data)
#to check if sample are correct
for sample in training_data[:10]:
    print(sample[1])
    
X=[]
y=[]

for features, label in training_data:
    X.append(features)    
    y.append(label)
#we cant pass a list to neurol networks, -1 means any number, 1 means grayscale
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y=np.array(y)
#we need to save our data for future use, and tweek it later
import pickle

pickle_out=open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in=open('X.pickle','rb')
X=pickle.load(pickle_in)
print(X[1])






















    
    
    
    
    
    
    
    
    
    