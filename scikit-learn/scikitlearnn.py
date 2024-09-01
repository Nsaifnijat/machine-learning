# -*- coding: utf-8 -*-

#features,attributes,independent variables, input, X
#label, dependent variable, output, y
#row is called dimension and col is called instances
from sklearn import joblib

#to save a model clf
filename='model.sav'
joblib.dump(clf,filename)
#to open
clf=joblib.load(filename)