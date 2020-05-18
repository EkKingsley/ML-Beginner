#using Linear Regression to predict the quality of wine
#data set is the winequality-white.csv
import pandas as pd
df = pd.read_csv('Deep-Learning/datasets/winequality-white.csv')
df.describe()

#plotting to help indicate if relationships exist b/n independent and dependent variables
import matplotlib.pylab as plt
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

#fitting and evaluating the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X = df[list(df.columns)[:-1]]
y = df['quality']

regressor = LinearRegression()
#using cross_val_score to perform cross validation using the provided data and estimator
#specified a 5 fold cross validation using the cv keyword argument i.e.:
#each instance will be randomly assigned to 1 of the 5 partitions and each partition
#used to train and test the model.
#the cross_val_score returns a value for the estimator's score method fr each round
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)
