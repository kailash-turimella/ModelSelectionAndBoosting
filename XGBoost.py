"""
XGBoost

Fast execution speed
You can keep the interpretation(Feature scaling is not required)
High performance

Suitable for large datasets
"""
# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data into dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder() 
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) # 2nd column
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2]) # 3rd column
# Avoiding dummy variable trap
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, # The Model
                             X = X_train,            # Independant variables
                             y = y_train,            # Dependant variable
                             cv = 10,                # 10 Accuracies (number of folds)
                             n_jobs = -1)            # Uses all the CPUs of your computer(not compulsory)
accuracies.mean()  # Average
accuracies.std()   # Standard Deviation(variance)

"""
https://colab.research.google.com/drive/
"""
