import cv2
import numpy as np

image = cv2.imread('./data/Lena.png').astype(np.float32) / 255
print('Shape:', image.shape)
print('Data type:', image.shape)
cv2.imshow('image', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape:', gray.shape)
print('Data type:', gray.shape)
cv2.imshow('image', gray)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape:', hsv.shape)
print('Data type:', hsv.shape)

hsv[:, :, 2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape:', from_hsv.shape)
print('Data type:', from_hsv.shape)
cv2.imshow('from_hsv', from_hsv)


cv2.waitKey()
cv2.destroyAllWindows()