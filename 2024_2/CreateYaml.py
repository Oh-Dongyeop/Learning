import yaml
import os

# 새로 생성할 데이터셋 폴더 구조 정의
dataset_folder = 'dataset'
train_folder = os.path.join(dataset_folder, 'train')
val_folder = os.path.join(dataset_folder, 'val')
test_folder = os.path.join(dataset_folder, 'test')

# YAML 파일 생성
yaml_data = {
    'train': os.path.abspath(train_folder),
    'val': os.path.abspath(val_folder),
    'test': os.path.abspath(test_folder),  # 선택 사항
    'nc': 3,  # 클래스 수
    'names': ['soldier1', 'soldier2', 'soldier3'],  # 클래스 이름
}

yaml_file = os.path.join(dataset_folder, 'rt-detr.yaml')
with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f)

print(f"YAML 파일이 생성되었습니다: {yaml_file}")
