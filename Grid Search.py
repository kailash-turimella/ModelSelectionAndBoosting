"""
Grid Search

Testing the model with different hyper-parameters 
  to help us determine the best value for the parameter
  
There are two types of parameters:
    The ones we choose (e.g. the kernal used to map the data onto a higher dimension)
    
    The ones the computer learns (e.g. weight coefficients in a linear regression model)
"""
#  Importing libraries
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
X_test = sc_X.transform(X_test)

# Fitting classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf') # Method used to map the data onto a higher dimension
classifier.fit(X_train,y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Applying K-Fold cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, # Machine Learning Model
                             X = X_train,            # Independant Variables
                             y = y_train,            # Dependant Variables
                             cv = 10)                # Number of folds
accuracies.mean()  # Average Accuracy
accuracies.std()   # Standard Deviation

# Applying grid search to find best model and the best parameters
from sklearn.model_selection import GridSearchCV
# C - helps us avoid overfitting
# Parameters of SVC
parameters = [{'C':[1,10,100,1000],                           # too high causes underfitting 
               'kernel':['linear']},                          # linear kernel(straight line)
              {'C':[1,10,100,1000],                           # too low causes overfitting
               'kernel':['rbf'],                              # The Gaussian RBF kernel(Non-linear)
               'gamma':[0.5,0.1,0.2,0.3,0.4,0.6,0.7,0.8]}]    # if 'auto', uses 1 / n_features
# Therefore we know a range of values for the gamma parameter

# list of dictionaries
grid_search = GridSearchCV(estimator=classifier,  # Our Machine Learning Model
                           param_grid=parameters, # The grid of parameters with different options 
                           scoring='accuracy',    # The evaluation of the prediction is based on the 'accuracy'
                           cv=10,                 # 10 fold cross validation
                           n_jobs=-1)             # Use all the CPU(only necessary for large datasets)

grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
