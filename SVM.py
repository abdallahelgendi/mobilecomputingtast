import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()             #load the data

#print(iris.data[:,:2])  #print the input data
#print(iris.target)      #print the output data

X = iris.data[:, :2]                     #put input  in X
y = iris.target                          #put output in y  output-->array contains 0s ,1 s or 2s

#getting the min and max in the two dimensions(fetures) values for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

#training the model
C = 1.0
Svm_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)

#plotting the predicted areas
Z = Svm_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)#for coloring the areas in the plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
plt.show()
