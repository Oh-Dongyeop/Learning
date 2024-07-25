import cv2
import numpy as np

# 이미지 로드
img = cv2.imread('./data/Lena.png')

# 임의의 노이즈 추가
noise = np.random.normal(0, 1, img.shape).astype(np.uint8)
noisy_img = cv2.add(img, noise)

# 필터링 알고리즘 적용
gaussian_filtered = cv2.GaussianBlur(noisy_img, (5, 5), 0)
median_filtered = cv2.medianBlur(noisy_img, 5)
bilateral_filtered = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# 결과 출력
cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Gaussian Filtered Image', gaussian_filtered)
cv2.imshow('Median Filtered Image', median_filtered)
cv2.imshow('Bilateral Filtered Image', bilateral_filtered)

# 입력 영상과 절대값 차이 취하기
gaussian_diff = cv2.absdiff(img, gaussian_filtered)
median_diff = cv2.absdiff(img, median_filtered)
bilateral_diff = cv2.absdiff(img, bilateral_filtered)

# 차이 결과 출력
cv2.imshow('Gaussian Difference Image', gaussian_diff)
cv2.imshow('Median Difference Image', median_diff)
cv2.imshow('Bilateral Difference Image', bilateral_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
