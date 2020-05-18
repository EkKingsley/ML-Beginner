#using sms+spam collection dataset
import pandas as pd

df = pd.read_csv('Deep-Learning/datasets/sms_spam.csv')
print(df.head())

print('Number of spam messages:', df[df[0] == 'spam'][0].count())
print('Number of ham messages:', df[df[0] == 'ham'][0].count())

#making some predictions using scikit-learn's LogisticRegression class:
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

#create a TfidfVectorizer that combines CountVectorizer an TfidfTransformer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

#create an instance of LogisticRegression and train model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
for i, prediction in enumerate(predictions[:5]):
    print('Prediction: %s. Message: %s' % (prediction, X_test_raw[i]))

#Binary Classification performance metrics using confusin matrix
from sklearn.metrics import confusion_matrix

y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#accuracy
from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print('Accuracy:', accuracy_score(y_true, y_pred))

#tuning models with grid search
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
