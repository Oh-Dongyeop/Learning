import os
import cv2
import matplotlib.pyplot as plt

def process_image(image):
    # 이미지 크기를 70%로 축소
    resized_image = cv2.resize(image, None, fx=0.7, fy=0.7)

    # 화사하게 만들기
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30)  # 밝기 증가
    processed_hsv = cv2.merge([h, s, v])
    brightened_image = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)

    # 채도를 높이기
    increased_saturation = cv2.convertScaleAbs(brightened_image, alpha=1, beta=1)

    # 살짝 블러 처리
    blurred_image = cv2.GaussianBlur(increased_saturation, (5, 5), 0)

    return blurred_image


# 테스트를 위한 이미지 로드
image_path = './data/1.jpg'
image = cv2.imread(image_path)

# 이미지 처리 함수 호출
processed_image = process_image(image)

# 결과 이미지를 디스크에 저장
image_dir = 'data'
output_file = 'processed_image.jpg'
cv2.imwrite(os.path.join(image_dir, output_file), processed_image)
print("Processed image saved at:", image_dir)

# 결과를 확인하기 위해 이미지를 보여줌
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', processed_image)

# 결과를 확인하기 위해 이미지를 Matplotlib으로 출력
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 처리된 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
plt.title('Processed Image')
plt.axis('off')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()