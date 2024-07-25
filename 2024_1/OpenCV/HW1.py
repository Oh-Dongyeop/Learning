import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('./data/Lena.png')

# 사용자로부터 채널 입력 받기
channel = input("Enter channel (R, G, B): ")

# 채널에 따른 히스토그램 평탄화
if channel == 'R':
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
elif channel == 'G':
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
elif channel == 'B':
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])

hist, bins = np.histogram(img, 256, range=(0, 255))
plt.fill(hist)
plt.xlabel('histogram')
plt.show()

# 결과 출력
cv2.imshow('Equalized Image', img)

# HSV 컬러 스페이스로 변경
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# V 채널에 대한 히스토그램 평탄화
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

# 다시 BGR 컬러 스페이스로 변경
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 결과 출력
cv2.imshow('HSV Equalized Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
