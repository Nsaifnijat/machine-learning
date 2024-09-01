# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('car.data')
#separating features and labels
X=data[[
        'buying',
        'maint',
        'safety'
        ]].values
y=data[['class']]
print(X,y)

#we need to convert the data in the above dataset into machine readable data which is numbers
le=LabelEncoder()

for i in range(len(X[0])):
    X[:,i]=le.fit_transform(X[:,i])
    
#y conversion

label_mapping={
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
    
    }
y['class']=y['class'].map(label_mapping)
y=np.array(y)


#create our model, n_neighbors=25, means it compares with 25 of its neighbors, weight='uniform', menas all neighbors are equal
#if weight='distance',  then they are judged based on the distance
#object
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
#separating training data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
#train the model
knn.fit(X_train, y_train)

#prediction
prediction=knn.predict(X_test)

accuracy=metrics.accuracy_score(y_test, prediction)

print('Predictions:',prediction)
print('Accuracy:',accuracy)

a=1727
print('acutal value:',y[a])
print('predicted value',knn.predict(X)[a])