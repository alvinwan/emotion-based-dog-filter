from scipy.linalg import solve
import numpy as np
import csv
import time

X = []
Y = []
Y_oh = []

def one_hot(y, num_classes=6):
    return np.eye(num_classes)[y]

t0 = time.time()

X_train, X_test = np.load('X_train.npy'), np.load('X_test.npy')
Y_train, Y_test = np.load('Y_train.npy'), np.load('Y_test.npy')
Y_oh_train, Y_oh_test = one_hot(Y_train), one_hot(Y_test)

t1 = time.time()
print('Finished loading data:', t1 - t0)

# featurize
W = np.random.normal(size=(X_train.shape[1], 2304))
A_train, A_test = X_train.dot(W), X_test.dot(W)

t2 = time.time()
print('Finished stacking data:', t2 - t1)

ATA, ATy = A_train.T.dot(A_train), A_train.T.dot(Y_oh_train)
I = np.eye(ATA.shape[0])
reg = 1e10
w = np.linalg.inv(ATA).dot(ATy)
w_ridge = np.linalg.inv(ATA + reg * I).dot(ATy)

t3 = time.time()
print('Finished solving:', t3 - t2)

# ols
Yhat_train = np.argmax(A_train.dot(w), axis=1)
print('(ols) Train Accuracy:', float(np.sum(Yhat_train == Y_train)) / Y_train.shape[0])
Yhat_test = np.argmax(A_test.dot(w), axis=1)
print('(ols) Test Accuracy:', float(np.sum(Yhat_test == Y_test)) / Y_test.shape[0])

# ridge
Yhat_train = np.argmax(A_train.dot(w_ridge), axis=1)
print('(ridge) Train Accuracy:', float(np.sum(Yhat_train == Y_train)) / Y_train.shape[0])
Yhat_test = np.argmax(A_test.dot(w_ridge), axis=1)
print('(ridge) Test Accuracy:', float(np.sum(Yhat_test == Y_test)) / Y_test.shape[0])

t4 = time.time()
print('Total time:', t4 - t0)
