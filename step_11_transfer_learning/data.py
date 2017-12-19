"""Creates a new dataset using your face."""

import numpy as np
import cv2
import glob


X = []
Y = []
for path in glob.iglob('images/me/*'):
    image = cv2.imread(path)
    resized = cv2.resize(image, (48, 48))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    X.append(np.ravel(gray))
    Y.append(1)

fer_X_train = np.load('X_train.npy')
fer_Y_train = np.load('Y_train.npy')

n = len(X)
indices = np.random.randint(0, fer_X_train.shape[0], n)
X.extend(fer_X_train[indices])
Y.extend([0] * n)

X = np.stack(X)
Y = np.stack(Y)

print('Saving X to me_X_train.npy %s' % str(X.shape))
np.save('me_X_train.npy', X)
print('Saving Y to me_Y_train.npy %s' % str(Y.shape))
np.save('me_Y_train.npy', Y)