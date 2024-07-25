import cv2
import numpy as np

def apply_otsu_threshold(image):
    # Otsu 이진화 적용
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def apply_adaptive_threshold(image):
    # 적응형 이진화 (중앙값)
    adaptive_binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_binary_image

def apply_morphological_operations(image, operation, iterations):
    # 모폴로지 연산 적용
    kernel = np.ones((3, 3), np.uint8)
    if operation == 'Erosion':
        result = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=iterations)
    elif operation == 'Dilation':
        result = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=iterations)
    elif operation == 'Opening':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'Closing':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        print("Invalid operation selection.")
        return None
    return result

# 입력 이미지 로드
image = cv2.imread('./data/Lena.png', cv2.IMREAD_GRAYSCALE)

# 이진화 방법 선택
method = input("이진화 방법을 선택하세요 (Otsu 또는 Adaptive): ")

# 선택된 방법에 따라 이진화 수행
if method == 'Otsu':
    binary_image = apply_otsu_threshold(image)
elif method == 'Adaptive':
    binary_image = apply_adaptive_threshold(image)
else:
    print("Invalid thresholding method selection.")
    exit()

# 결과 이미지 표시
cv2.imshow('Binary Image', binary_image)

# 모폴로지 연산 선택
operation = input("적용할 모폴로지 연산을 선택하세요 (Erosion, Dilation, Opening, Closing): ")

# 선택된 모폴로지 연산에 따라 횟수 입력
if operation in ['Erosion', 'Dilation', 'Opening', 'Closing']:
    iterations = int(input("모폴로지 연산을 수행할 횟수를 입력하세요: "))
else:
    print("Invalid operation selection.")
    exit()

# 모폴로지 연산 적용
morphological_result = apply_morphological_operations(binary_image, operation, iterations)

# 결과 이미지 표시
cv2.imshow('Morphological Result', morphological_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
