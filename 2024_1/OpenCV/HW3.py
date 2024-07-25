import cv2
import numpy as np

# 이미지 로드
img = cv2.imread('./data/Lena.png', cv2.IMREAD_GRAYSCALE)

# DFT 수행
fft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])

# 보다 잘 시각화하기 위해 크기 스펙트럼을 로그 스케일로 변환
magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1]))


# 사용자로부터 내부 반지름과 외부 반지름 입력 받기
radius_inner = int(input("내부 반지름 입력: "))
radius_outer = int(input("외부 반지름 입력: "))

# 영상의 중심 좌표
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Band Pass 필터 생성
mask = np.zeros(fft.shape, np.uint8)
mask = cv2.circle(mask, (ccol, crow), radius_inner, 1, -1)  # 내부 원
mask = cv2.circle(mask, (ccol, crow), radius_outer, 0, -1)  # 외부 원

# 마스크를 DFT 이미지에 적용
filtered_dft = fft * mask

# 역 DFT 수행
filtered_image = cv2.idft(filtered_dft)

# 크기 스펙트럼 계산
filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

# 결과를 적절히 표시하기 위해 정규화
cv2.normalize(filtered_image, filtered_image, 0, 255, cv2.NORM_MINMAX)
filtered_image = filtered_image.astype(np.uint8)

# 결과 출력
cv2.imshow('Input Image', img)
# 크기 스펙트럼 출력
cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
cv2.imshow('Band Pass Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
