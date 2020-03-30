import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()             #load the data
X = iris.data[:, :4]                     #put input  in X
y = iris.target                          #put output in y  output-->array contains 0s ,1 s or 2s

#preparing training,testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#training the model
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4), random_state=1)#one hidden layer with 4 neurons
classifier.fit(X_train, y_train)

#making predictions to test the model
y_pred = classifier.predict(X_test)

#printing the weights
print("the coefs are : ",classifier.coefs_)

#printing the bias
print("\n the biases are : ",classifier.intercepts_)
   

#calculate the number of correct predictions
size=y_pred.size
count=0
for i in range(size):
    if y_test[i]==y_pred[i]:
        count+=1

#print the accuracy  (the number of correct predictions/the num of all predictions)
print("\n the accuracy = ",count/size) 