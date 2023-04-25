import numpy as np
from sklearn.neural_network import MLPClassifier
from ReadImages import ReadImages
from sklearn.svm import SVC

# Reading Data from file
Factory = ReadImages("/Users/david_da_lian/Downloads/archive")
Factory.Reading()
Factory.Split()
print("finished")

# Splitting Data in to 811
X_train, y_train = Factory.get_training()
X_test, y_test = Factory.get_testing()
X_val, y_val = Factory.get_validation()
print(len(X_train))
print(len(y_train))
print(X_train.shape)

# Build First Classifier (MLPC) --------------------------------------------------
print('MLPC:')
clf_mlpc = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42)
clf_mlpc.fit(X_train, y_train)

# Evaluate the classifier on the training data
print("Training Accuracy: {}".format(clf_mlpc.score(X_train, y_train)))

# Evaluate the classifier on the validation data
print("Validation Accuracy: {}".format(clf_mlpc.score(X_val, y_val)))

# Evaluate the classifier on the testing data
print("Testing Accuracy: {}".format(clf_mlpc.score(X_test, y_test)))

# Build Second Classifier (SVM) --------------------------------------------------
print('SVM:')
clf_svm = SVC(kernel='rbf', C=1)
clf_svm.fit(X_train, y_train)

# Evaluate the classifier on the training data
print("Training Accuracy: {}".format(clf_svm.score(X_train, y_train)))

# Evaluate the classifier on the testing data
print("Testing Accuracy: {}".format(clf_svm.score(X_test, y_test)))

# Find Better hyperparamerter for SVM --------------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# define the parameter grid
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

# instantiate an SVM classifier
svc = SVC()

# instantiate the grid search with a cross-validation of 5
grid_search = GridSearchCV(svc, param_grid, cv=5)

# fit the grid search to the training data
grid_search.fit(X_train, y_train)

# select the best hyperparameters
best_params = grid_search.best_params_

# train the model on the entire training set using the best hyperparameters
clf = SVC(**best_params)
clf.fit(X_val, y_val)

# evaluate the performance on the validation set
y_pred_val = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", val_accuracy)

# evaluate the performance on the test set
y_pred_test = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", test_accuracy)