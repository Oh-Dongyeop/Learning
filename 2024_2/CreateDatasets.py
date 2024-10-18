import os
import random
import shutil

# 원본 데이터 폴더 경로 설정 (이미지와 라벨이 섞여 있는 폴더)
source_folder = 'soldier'

# 새로 생성할 데이터셋 폴더 구조 정의
dataset_folder = 'dataset'
train_folder = os.path.join(dataset_folder, 'train')
val_folder = os.path.join(dataset_folder, 'val')
test_folder = os.path.join(dataset_folder, 'test')

# 폴더 생성
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 이미지 파일과 라벨 파일 목록 가져오기
files = os.listdir(source_folder)
images = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
labels = [f for f in files if f.endswith('.txt')]

# 이미지와 라벨 매칭
data = []
for image in images:
    label = image.replace('.jpg', '.txt').replace('.png', '.txt')
    if label in labels:
        data.append((image, label))

# 데이터 셔플링
random.shuffle(data)

# 학습/검증/테스트 데이터 비율 설정
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 데이터셋 나누기
total_data = len(data)
train_data = data[:int(total_data * train_ratio)]
val_data = data[int(total_data * train_ratio):int(total_data * (train_ratio + val_ratio))]
test_data = data[int(total_data * (train_ratio + val_ratio)):]

# 데이터 복사 함수 (이미지와 라벨 파일을 동일한 폴더에 복사)
def copy_data(data_split, dest_folder):
    for image_file, label_file in data_split:
        # 이미지 파일 복사
        shutil.copy(os.path.join(source_folder, image_file), os.path.join(dest_folder, image_file))
        # 라벨 파일 복사
        shutil.copy(os.path.join(source_folder, label_file), os.path.join(dest_folder, label_file))

# 학습, 검증, 테스트 데이터 복사
copy_data(train_data, train_folder)
copy_data(val_data, val_folder)
copy_data(test_data, test_folder)

print("이미지와 라벨 파일이 동일한 폴더에 train/val/test로 나누어졌습니다.")

