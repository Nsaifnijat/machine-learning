# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
#split it in features and labels

X=iris.data
y=iris.target

print(X.shape)
print(y.shape)

#explanation in an example
#lets train a model, it has 10 students
#good/bad grades judged based on the hours of study
#train the model with 8 students
#let the model predict the grades of the other 2 students

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#test_size=0.2 means 20 percent of data is for testing purpose
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#KNN, K nearest neighbors