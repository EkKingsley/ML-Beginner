#Using linear regression model to predict the prize of pizza given
#the diameter of the pizza in inches

#Simple Linear Regression
import matplotlib.pyplot as plt
X = [[6], [8], [10], [14], [18]] #diameter of pizza in inches
y = [[7], [9], [13], [17.5], [18]] #price of pizza in dollars

#visualizing the training data by plotting in on a graph using matplotlib:
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in Inches')
plt.ylabel('Price in Dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

#modelling the relationship using linear regression:
from sklearn.linear_model import LinearRegression

#training data
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

#create and fit the model
model = LinearRegression() #creates model
model.fit(X, y) # fit() trains model on the training data X and y
print('A 12" pizza should cost: $%.2f' % model.predict([[12]])[0]) #predict() for making predictions with the trained model

#evaluating the model's predictive performance using score()
#R-squared measures how well the observed values of the response variables
#are predicted by the model
#R-squaared score of 1 indicates prediction without any error using the model
#R-squared score of half indicates that half of the variance in the response variabel cn be predicted using the model

#test sets
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]

print('R-spuared: %.4f' % model.score(X_test, y_test))

#Multiple Linear Regression
#Updating the pizza-price predictor to use a second explanatory variable,
#toppings, and compare its performance on the test set to that of the
#simple linear regression model:
from sklearn.linear_model import LinearRegression
#training sets
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]

#create model
model = LinearRegression()

#fit the model to the training sets
model.fit(X, y)

#test sets
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(X_test) #make predictions on the X_test

for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R-square: %.2f' % model.score(X_test, y_test))

#Polynomial regression
#used as its not always true that in the real relationship b/n the
#explanatory variables and the response variable is linear.
#In this section, polynomial regression, a special case of multiple
#linear regression that adds terms with degrees greater than 1 to the
#model.
#The PolynomialFeatures transformer cn be used to easily add polynomial features
#to a feature representation.
#Let's fit a model to these features, and compare it to the simple linear regression model:

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
#creating and adding polynomial features to the X_train and X_test sets
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

#fit the quadratic train and test set to a linear regression model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)

xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in Inches')
plt.ylabel('Price in Dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('Simple Linear r-squared', regressor.score(X_test, y_test))
print('Quadratic Regression r-squared', regressor_quadratic.score(X_test_quadratic, y_test))

#using stochastic gradient descend to estimate the parameters of a model.
#the SGDRegressor is an implementation of SGD
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)

X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('Cross Validation r-squared scores:', scores)
print('Average cross validation r-squared score:', np.mean(scores))

regressor.fit_transform(X_train, y_train)
print('Test set r-squared score', regressor.score(X_test, y_test))
