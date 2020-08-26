"""
K-Fold cross validation

Used to evaluating the performance of our machine learning model

Splitting the training set into a number of folds(10)
Training the model on 9 folds and testing it on the 10th fold
"""
# Importing libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


# Importing datasets
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)          # Already fitted

# Fitting classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf') # Kernal(Method used to map the data onto a higher dimension)
classifier.fit(X_train,y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Applying K-Fold cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, # The Model
                             X = X_train,            # Independant variables
                             y = y_train,            # Dependant variable
                             cv = 10,                # 10 Accuracies (number of folds)
                             n_jobs = -1)            # Uses all the CPUs of your computer(not compulsory)
accuracies.mean()  # Average
accuracies.std()   # Standard Deviation(variance)

# mean - std < Accuracy < mean + std