import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread('./data/Lena.png', 0)
cv2.imshow('Original grey', gray)
cv2.waitKey()

hist, bins = np.histogram(gray, 256, range=(0, 255))
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

grey_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(grey_eq, 256, range=(0, 255))
plt.fill_between(range(256), hist,0)
plt.xlabel('pixel value')
plt.show()


cv2.imshow('equalized grey', grey_eq)
cv2.waitKey()

color = cv2.imread('./data/Lena.png')
hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
cv2.imshow('original color', color)
cv2.imshow('equalized color', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()

