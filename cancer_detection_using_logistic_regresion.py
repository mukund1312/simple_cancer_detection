# importing all the libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pylab as mlt


# importing Data set and then slitting them into dependent and independent variable
Data_set = pd.read_csv("Clasification\\breast-cancer-wisconsin.csv")
x = Data_set.iloc[:, 1:-1].values
y = Data_set.iloc[:, -1].values

# now lets split them into test and train set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=0)


# training our models using logistic regresion on our train Set

clasi = LogisticRegression(random_state=0)
clasi.fit(x_train, y_train)

# predict the test set
y_pred = clasi.predict(x_test)

# lets now make the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# now lets see the accuracy
ac = accuracy_score(y_test, y_pred)
print(ac)


# Kfold cross validation
accuracies = cross_val_score(estimator=clasi, X=x_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
