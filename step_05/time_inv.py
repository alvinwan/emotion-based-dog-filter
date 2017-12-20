import numpy as np
import time

times = []

for d in range(20):
    start = time.time()
    print('Dimension:', 2**d)
    np.linalg.inv(np.random.random(size=(2**d, 2**d)))
    end = time.time()
    times.append(end - start)
    print(end - start)

print(times)
