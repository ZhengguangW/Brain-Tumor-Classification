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