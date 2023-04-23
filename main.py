import numpy as np
from sklearn.neural_network import MLPClassifier
from ReadImages import ReadImages

Factory=ReadImages("/Users/zhengguang/Desktop/OneDrive - University of Virginia/Desktop/CS 4774/Brain-Tumoer-Classification/archive")
Factory.Reading()
Factory.Split()
print("finished")
X_train,y_train=Factory.get_training()
print(len(X_train))
print(len(y_train))
X_test,y_test=Factory.get_testing()
print(X_train.shape)
Factory.get_validation()
clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42)
clf.fit(X_train,y_train)

# Evaluate the classifier on the testing data
score = clf.score(X_test, y_test)
print("Accuracy:", score)