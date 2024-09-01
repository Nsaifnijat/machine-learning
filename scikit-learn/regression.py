# -*- coding: utf-8 -*-

import pandas as pd
#import quandl
import  math,datetime

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
df=pd.read_csv('gooogle.csv',index_col='Date',parse_dates=[0])
#addding and calculating the following columns to the imported df
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0


#separating useful features from the above data for our model
df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#making a variable
forecast_col='Adj. Close'

#you cant work with nan values in ML, either you remove or fill them
df.fillna(-99999,inplace=True)

#number of days you wanna predict using the past data
forecast_out=int(math.ceil(0.1*len(df)))


#now we have features,lets create labels,label is created from 'Adj. Close' values but shifted to the number of forecast_out from the end of the column to the past
df['label']=df[forecast_col].shift(-forecast_out)

#df.drop(['label],1) it returns a new dataframe in which label is droped
X=np.array(df.drop(['label'],1))

X=preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

df.dropna(inplace=True)
y=np.array(df['label'])
#create our training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
'''
#create model,you can thread your regression by specifying n_jobs=10, which makes ten threads and processing time is faster,
#or put n_jobs=-1, which makes as many as threads as your process allows, default is 1
clf=LinearRegression(n_jobs=-1)
#to shift your algorithm to another just change the above
#clf=svm.SVR()
or svm with a kernal
#clf=svm.SVR(kernal='poly')

#train
clf.fit(X_train,y_train)
#we save our trained model to avoid time consumption
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f) '''
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)
#test
accuracy=clf.score(X_test,y_test)

forecast_set=clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast']=np.nan

#getting the last date
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day
print(last_date)
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()









