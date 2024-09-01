# -*- coding: utf-8 -*-

import pandas as pd
import quandl, math,datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')
#df=quandl.get('WIKI/GOOGL')
#df.to_csv('gooogle.csv')
df=pd.read_csv('data/masterframe.csv',index_col='date',parse_dates=[0])
print(df.shape)

#you cant work with nan values in ML, either you remove or fill them
df.fillna(-99999,inplace=True)

#number of days you wanna predict using the past data
forecast_out=10


#now we have features,lets create labels,label is created from 'Adj. Close' values but shifted to the number of forecast_out from the end of the column to the past
df['label']=df['close'].shift(-forecast_out)
df.dropna(inplace=True)

#df.drop(['label],1) it returns a new dataframe in which label is droped
X=np.array(df.drop(['label','close'],1))
X=preprocessing.scale(X)
print(X.shape)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

y=np.array(df['label'])
y_lately=y[-forecast_out:]
y=y[:-forecast_out]
y.shape
#create our training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

#create model,you can thread your regression by specifying n_jobs=10, which makes ten threads and processing time is faster,
#or put n_jobs=-1, which makes as many as threads as your process allows, default is 1
clf=LinearRegression(n_jobs=-1)
#to shift your algorithm to another just change the above
#clf=svm.SVR()
#or svm with a kernal
#clf=svm.SVR(kernal='poly')

#train
clf.fit(X_train,y_train)
'''
#we save our trained model to avoid time consumption
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f) 
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)'''
#test
accuracy=clf.score(X_test,y_test)
clf.coef_
clf.intercept_
forecast_set=clf.predict(X_lately)

accuracy=clf.score(X_lately,y_lately)

print(accuracy)


'''

Step 1: Set a goal
Let’s assume our goal is to understand the relationship between interest rates and EUR/USD exchange rate and how strongly correlated these two variables are.
Our dependent variable is: Interest Rates as it is dependent on EUR/USD exchange rates. Dependent variable is known as regressand in statistics.
Our independent variable is: EUR/USD exchange rates. Independent variable is known as regressor in statistics.
Intercept — where line crosses y axis — Known as ɑ (ALPHA)
Slope — Tell us expected change in y over unit change in x — β (BETA)
Distance between actual data point and the best fit data point — ε (RESIDUAL)


With linear regression, fitting the model means determining the best intercept (model.intercept_) and 
slope (model.coef_) values of the regression line.





'''






