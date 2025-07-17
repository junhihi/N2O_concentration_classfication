import pandas as pd
import os
import re
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
import torch
from itertools import repeat

# 디바이스
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 전역 transform 정의
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_config(config_path='config/config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def extract(data, t0, t1):
    """지정된 시간 범위 내 데이터 추출"""
    return data[(t0 <= data['time[ms]']) & (data['time[ms]'] <= t1)][['base', '기준신호 1']]

def to_jpg(data, file, output_dir='dataset/imgs', augmentation=True):
    """데이터를 JPG 이미지로 변환"""
    folder = os.path.join(output_dir, file)
    os.makedirs(folder, exist_ok=True)

    n, start = 0, 0
    samples = list()
    while True:
        try:
            sample = os.path.join(folder, f'sample_{n}.jpg')
            start = extract(data, n, n+1)['기준신호 1'].idxmin()
            end = start + 10000
            if end > len(data):
                break
            cycle = data[start:end].reset_index(drop=True)
            plt.figure(figsize=(10, 10), dpi=100)
            plt.plot(cycle['기준신호 1'])
            plt.plot(cycle['base'])
            plt.axis('off')
            plt.savefig(sample, bbox_inches='tight')
            plt.close()
            n += 1
            samples.append(sample)
            if not augmentation:
                break
        except ValueError:
            break
    return folder, samples

def labeling(folder_path='dataset/csv_files', output_dir='dataset/imgs'):
    """데이터셋 라벨링 및 이미지 경로 생성"""
    folders = [f.split('.')[0] for f in os.listdir(folder_path) if f.endswith('.csv')]
    datas = [pd.read_csv(os.path.join(folder_path, f'{f}.csv'), encoding='cp949') for f in folders]

    with ProcessPoolExecutor() as executor:
        new_folders = list(executor.map(to_jpg, datas, folders, repeat(output_dir)))
    
    img = defaultdict(list)
    for f, s in new_folders:
        img[torch.tensor(float(re.search(r'_(\d+)%', f).group(1))/100, dtype=torch.float32)].extend(s)

    return img