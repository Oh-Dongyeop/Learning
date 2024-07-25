import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('./data/Lena.png', 0)
cv2.imshow('Original grey', gray)
cv2.waitKey()

grey_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(grey_eq, 256, range=(0, 255))
plt.fill_between(range(256), hist,  0)
plt.xlabel('pixel value')
plt.show()

