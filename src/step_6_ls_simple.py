"""Simple least squares classifier for face emotion recognition.

Additionally uses random Gaussian matrix as featurization
"""

import numpy as np


def evaluate(A, Y, w):
    Yhat = np.argmax(A.dot(w), axis=1)
    return float(np.sum(Yhat == Y)) / Y.shape[0]

# load data
X_train, X_test = np.load('data/X_train.npy'), np.load('data/X_test.npy')
Y_train, Y_test = np.load('data/Y_train.npy'), np.load('data/Y_test.npy')

# one-hot labels
I = np.eye(3)
Y_oh_train, Y_oh_test = I[Y_train], I[Y_test]

# generate random Gaussian and featurize X
d = 100
W = np.random.normal(size=(X_train.shape[1], d))
A_train, A_test = X_train.dot(W), X_test.dot(W)

# train model
w = np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T).dot(Y_oh_train)

# evaluate model
ols_train_accuracy = evaluate(A_train, Y_train, w)
print('(ols) Train Accuracy:', ols_train_accuracy)
ols_test_accuracy = evaluate(A_test, Y_test, w)
print('(ols) Test Accuracy:', ols_test_accuracy)
