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
    for row in reader:
        y = int(row[0]) - 1
        if y not in (0, 3, 4):
            continue
        x = np.array(list(map(int, row[1].split())))
        X.append(x)
        Y.append(y)

t1 = time.time()
print('Finished loading data:', t1 - t0)

n_train = int(len(X) * 0.8)

X_train, X_test = np.array(X[:n_train]), np.array(X[n_train:])
Y_train, Y_test = np.array(Y[:n_train]), np.array(Y[n_train:])

np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('Y_train', Y_train)
np.save('Y_test', Y_test)

t2 = time.time()
print('Finished converting data')
