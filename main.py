import numpy as np
from sklearn.neural_network import MLPClassifier
from ReadImages import ReadImages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Reading Data from file
Factory = ReadImages("/Users/zhengguang/Library/CloudStorage/OneDrive-UniversityofVirginia/Desktop/CS 4774/Brain-Tumoer-Classification/archive")
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

# # Build First Classifier (MLPC) --------------------------------------------------
# print('MLPC:')
# clf_mlpc = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42)
# clf_mlpc.fit(X_train, y_train)
#
# # Evaluate the classifier on the training data
# print("Training Accuracy: {}".format(clf_mlpc.score(X_train, y_train)))
#
# # Evaluate the classifier on the validation data
# print("Validation Accuracy: {}".format(clf_mlpc.score(X_val, y_val)))
#
# # Evaluate the classifier on the testing data
# print("Testing Accuracy: {}".format(clf_mlpc.score(X_test, y_test)))

# Build Second Classifier (SVM) --------------------------------------------------
print('SVM:')
clf_svm = SVC(kernel='poly', C=0.2, gamma=0.1)
clf_svm.fit(X_train, y_train)

# Evaluate the classifier on the training data
print("Training Accuracy: {}".format(clf_svm.score(X_train, y_train)))

# Evaluate the classifier on the testing data
print("Testing Accuracy: {}".format(clf_svm.score(X_test, y_test)))

# Define the hyperparameters to search over
# c_values = [0.1, 1, 10]
# gamma_values = [0.01, 0.1, 1]
# kernel_values = ['poly', 'rbf', 'sigmoid']

# # Create empty lists to store the results
# best_accuracy = 0
# best_params = {}

# for c in c_values:
#     for gamma in gamma_values:
#         for kernel in kernel_values:
#             # Create the SVM classifier with the current hyperparameters
#             clf_svm = SVC(kernel=kernel, C=c, gamma=gamma)

#             # Train the classifier on the training set
#             clf_svm.fit(X_train, y_train)

#             # Evaluate the classifier on the validation set
#             y_pred = clf_svm.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)

#             # Print out the current hyperparameters and their validation accuracy
#             print("Hyperparameters: C={}, gamma={}, degree={}".format(kernel, c, gamma))
#             print("Validation accuracy: {}".format(accuracy))

#             # Update the best hyperparameters if the current model is better
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_params = {'kernel': kernel, 'C': c, 'gamma': gamma}

# # Print out the best hyperparameters and their validation accuracy
# print("Best hyperparameters: ", best_params)
# print("Validation accuracy: ", best_accuracy)

# # Create the SVM classifier with the best hyperparameters
# clf_svm = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])

# # Train the classifier on the training set
# clf_svm.fit(X_train, y_train)

# # Evaluate the classifier on the test set
# y_pred = clf_svm.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# # Print out the test accuracy
# print("Test accuracy: ", accuracy)












activation = ["logistic","tanh","relu"]
hidden_layer_size=[(50,),(100,),(150,)]
learning_rate = ["constant","invscaling","adaptive"]

# Create empty lists to store the results
best_accuracy = 0
best_params = {}

for indiv_activation in activation:
    for indiv_layer_size in hidden_layer_size:
        for rate in learning_rate:
            # Create the SVM classifier with the current hyperparameters
            clf_mlpc = MLPClassifier(activation=indiv_activation,hidden_layer_sizes=indiv_layer_size,learning_rate=rate)

            # Train the classifier on the training set
            clf_mlpc.fit(X_train, y_train)

            # Evaluate the classifier on the validation set
            y_pred = clf_mlpc.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            # Print out the current hyperparameters and their validation accuracy
            print("Hyperparameters: indiv_activation={}, indiv_layer_size={}, rate={}".format(indiv_activation, indiv_layer_size, rate))
            print("Validation accuracy: {}".format(accuracy))

            # Update the best hyperparameters if the current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"Activation":indiv_activation,'layer_size': hidden_layer_size, "learning rate":rate}

# Print out the best hyperparameters and their validation accuracy
print("Best hyperparameters: ", best_params)
print("Validation accuracy: ", best_accuracy)

