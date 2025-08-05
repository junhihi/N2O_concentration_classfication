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
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# 전역 transform 정의
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
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

    n, cnt = 0, 0
    samples = list()
    sampling = 10000
    aug = 10
    while True:
        try:
            s_ref = extract(data, n, n+1)['기준신호 1'].idxmin()
            e_ref = s_ref+sampling
            if e_ref > len(data):
                    break
            for i in range(aug):
                sample = os.path.join(folder, f'sample_{cnt}.jpg')
                start = s_ref+int(sampling/aug)*i
                end = start + sampling
                if end > len(data):
                    break
                cycle = data[start:end].reset_index(drop=True)
                # min-max 정규화
                cycle['base'] = (cycle['base'] - cycle['base'].min()) / (cycle['base'].max() - cycle['base'].min())
                # 표준화
                # cycle['base'] = (cycle['base'] - cycle['base'].mean()) / cycle['base'].std()
                plt.figure(figsize=(5, 5), dpi=100)
                # plt.plot(cycle['기준신호 1'])
                plt.plot(cycle['base'], linewidth=.3, c='black')
                plt.axis('off')
                plt.savefig(sample, bbox_inches='tight')
                plt.close()
                samples.append(sample)
                cnt += 1
                if not augmentation:
                    break
            n += 1
        except ValueError:
            break
    return folder, samples


def labeling(csv_path='dataset/csv_files', output_dir='dataset/imgs', augmentation=True):
    """데이터셋 라벨링 및 이미지 경로 생성"""
    if not os.path.exists(csv_path) or not os.path.exists(output_dir):
        os.makedirs(csv_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    op_dir = [f for f in os.listdir(output_dir) if not f.endswith('.DS_Store')]
    csv_files = [c.split('.')[0] for c in os.listdir(csv_path) if c.endswith('.csv')]
    
    new_csv = [c for c in csv_files if c not in op_dir]
    datas = [pd.read_csv(os.path.join(csv_path,f'{c}.csv'), encoding='cp949') for c in new_csv]

    if new_csv:
        with ProcessPoolExecutor() as executor:
            new_folders = list(executor.map(to_jpg, datas, new_csv, repeat(output_dir), repeat(augmentation)))
    
    else:  
        new_folders = [
            (
                os.path.join(output_dir, folder),
                [
                    os.path.join(output_dir, folder, file)
                    for file in os.listdir(os.path.join(output_dir, folder))
                    if not file.startswith('.DS_Store')
                ]
            )
            for folder in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, folder)) and not folder.startswith('.DS_Store')
        ]

    img = defaultdict(list)
    for f, s in new_folders:
        img[torch.tensor(float(re.search(r'_(\d+)%', f).group(1))/100, dtype=torch.float32)].extend(s)

    return img