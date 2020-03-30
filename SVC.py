import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names = headernames)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 4].values
print(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
C = 1.0
Svc_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)
y_pred = Svc_classifier.predict(X_test)

size=y_pred.size
count=0
for i in range(size):
    if y_test[i]==y_pred[i]:
        count+=1
    
print("the accuracy = ",count/size)

