# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston=datasets.load_boston()

#features/labels
x=boston.data
y=boston.target

#print(x)
#print(x.shape)
#print('Y',y)
#algorithm
l_reg=linear_model.LinearRegression()

#plt.scatter(x.T[4],y)
#plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

#train
model=l_reg.fit(x_train,y_train)
prediction=model.predict(x_test)

print('predictions:',prediction)
print('R^2 values:',l_reg.score(x,y))
print('Coeff:',l_reg.coef_)
print('intercept:',l_reg.intercept_)