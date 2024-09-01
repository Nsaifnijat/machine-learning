# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
#split it in features and labels

X=iris.data
y=iris.target

classes=['Iris Setosa','Iris Versicolour','Iris Virginica']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#test_size=0.2 means 20 percent of data is for testing purpose
model=svm.SVC()
model.fit(X_train,y_train)


prediction=model.predict(X_test)

accuracy=accuracy_score(y_test, prediction)

print('Predictions:',prediction)
print('acutal value:',y_test)
print('Accuracy:',accuracy)
print(model)

#to print the names of the predictions
for i in range(len(prediction)):
    print(classes[prediction[i]])