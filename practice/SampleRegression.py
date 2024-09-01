# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

x, y = load_boston(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.4, random_state=0)


#using the following regression models and testing their accuracy

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
model.score(x_train, y_train)
print('regrssion train',model.score(x_train, y_train))
model.score(x_test, y_test)
print('regrssion test',model.score(x_test, y_test))



from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
model.score(x_train, y_train)
print('GBR train',model.score(x_train, y_train))
model.score(x_test, y_test)
print('GBR test',model.score(x_test, y_test))


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
model.score(x_train, y_train)
print('RFR train',model.score(x_test, y_test))
model.score(x_test, y_test)
print('RFR test',model.score(x_test, y_test))














