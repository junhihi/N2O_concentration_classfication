from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class CustomDataset(Dataset):
    """이미지 데이터셋 클래스"""
    def __init__(self, image_paths, labels, transform=None):
        """이미지 경로와 라벨로 데이터셋 초기화"""
        valid_pairs = [(path, label) for path, label in zip(image_paths, labels) if os.path.exists(path)]
        if not valid_pairs:
            raise ValueError(f"유효한 이미지 경로가 없습니다. 샘플 경로: {image_paths[:5]}")
        self.image_paths = [pair[0] for pair in valid_pairs]
        self.labels = [pair[1] for pair in valid_pairs]
        self.transform = transform
        print(f"데이터셋 초기화 완료: {len(self.image_paths)} 샘플")

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """인덱스에 해당하는 이미지와 라벨 반환"""
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"이미지 로드 오류: {self.image_paths[idx]} - {e}")
            return torch.zeros((1, 224, 224)), 0