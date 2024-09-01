# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

df=pd.read_excel('titanic.xls')
#to check which column has na vals
#print(df.isna().sum())
#to exclude na vals from a column in the dataframe
df=df.loc[df.embarked.notna(),['survived','pclass','sex','embarked']]

#feature or x , we get only pclass
x=df.drop('survived',1)
y=df.survived

column_trans=make_column_transformer(
    (OneHotEncoder(),['sex','embarked']),
    remainder='passthrough')

logreg=LogisticRegression(solver='lbfgs')

#to get all the parameters a model has, here of the estimator or model logisticregression
logreg.get_params()

pipe=make_pipeline(column_trans, logreg)
#5 fold cross validation and getting the mean accuracy of 5 fold validation
cross_val_score(pipe, x,y, cv=5,scoring='accuracy').mean()

#just gonna make a small test set from the training data, 5 rows
x_new=x.sample(5,random_state=99)

#train
pipe.fit(x,y)

pipe.predict(x_new)
































