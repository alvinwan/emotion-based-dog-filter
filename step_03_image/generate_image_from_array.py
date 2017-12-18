import numpy as np
import cv2

string = """0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 1 1 1 1 1 0
0 0 1 1 1 0 0
0 0 0 1 0 0 0
0 0 0 0 0 0 0"""

array = np.array([
    list(map(int, line.split())) for line in string.split('\n')
]) * 255.

larger = cv2.resize(array, None, fx=60, fy=60, interpolation=cv2.INTER_AREA)

cv2.imwrite('diamond.jpg', larger)
