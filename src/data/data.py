"""Data conversion script, from csv to numpy files.

Face Emotion Recognition dataset by Pierre-Luc Carrier and Aaron Courville

Keeps only 3 classes, where the new class indices are 0, 1, 2, respectively:
3 - happy
4 - sad
5 - surprise
"""

from scipy.linalg import solve
import numpy as np
import csv
import time

X = []
Y = []

t0 = time.time()

with open('fer2013/fer2013.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, row in enumerate(reader):
        y = int(row[0])
        if y not in (3, 4, 5):
            continue
        y -= 3
        x = np.array(list(map(int, row[1].split())))
        X.append(x)
        Y.append(y)

t1 = time.time()
print('Finished loading data:', t1 - t0)

n_train = int(len(X) * 0.8)

X_train, X_test = np.array(X[:n_train]), np.array(X[n_train:])
Y_train, Y_test = np.array(Y[:n_train]), np.array(Y[n_train:])

np.save('X_train', X_train)
print('Saved X_train %s' % str(X_train.shape))
np.save('X_test', X_test)
print('Saved X_test %s' % str(X_test.shape))
np.save('Y_train', Y_train)
print('Saved Y_train %s' % str(Y_train.shape))
np.save('Y_test', Y_test)
print('Saved Y_test %s' % str(Y_test.shape))

t2 = time.time()
print('Finished converting data')
