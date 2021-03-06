import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#getting the data
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names = headernames)

#defining input and output
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values
print(dataset)

#defining training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#training the model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#making prediction to test the model 
y_pred = classifier.predict(X_test)

#calculating the accuracy
size=y_pred.size
count=0
for i in range(size):
    if y_test[i]==y_pred[i]:
        count+=1
 
#print the accuracy  (the number of correct predictions/the num of all predictions)
print("the accuracy = ",count/size)
