import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import os
from utils import TRANSFORM, load_config, labeling, get_device
from dataset import CustomDataset
from model import CNN
import yaml
import numpy as np  # RMSE 계산을 위해 추가

def collate_fn(batch):
    '''배치에서 None 값을 가진 샘플을 필터링'''
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

class ModelTrainer:
    """모델 학습 및 평가 클래스"""
    # 나중에 config에서 device, augmentation 추가
    def __init__(self, model, config):
        self.device = get_device()
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.checkpoint_path = config['checkpoint_path']
        logging.basicConfig(filename=config['log_path'], level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            logging.info(f'Checkpoint loaded from {self.checkpoint_path}')
            print(f"체크포인트 로드 완료: {self.checkpoint_path}")

    def calculate_metrics(self, loader):
        """데이터 로더에서 손실과 MAE, RMSE 계산"""
        self.model.eval()
        total_loss, total_mae, total_mse, total = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                if inputs is None or labels is None or inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device).view(-1, 1)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item()
                total_mae += torch.mean(torch.abs(outputs - labels)).item()
                total_mse += torch.mean((outputs - labels) ** 2).item()
                total += labels.size(0)
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        mae = total_mae / len(loader) if len(loader) > 0 else 0
        rmse = np.sqrt(total_mse / len(loader)) if len(loader) > 0 else 0
        return avg_loss, mae, rmse

    def train(self, train_loader, val_loader, num_epochs):
        """모델 학습"""
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in train_loader:
                if inputs is None or labels is None or inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device).view(-1, 1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            train_loss, train_mae, train_rmse = self.calculate_metrics(train_loader)
            val_loss, val_mae, val_rmse = self.calculate_metrics(val_loader)
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')
            print(f'에포크 {epoch+1}, 학습 손실: {train_loss:.4f}, 학습 MAE: {train_mae:.4f}, 학습 RMSE: {train_rmse:.4f}, '
                  f'검증 손실: {val_loss:.4f}, 검증 MAE: {val_mae:.4f}, 검증 RMSE: {val_rmse:.4f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save({'model_state_dict':self.model.state_dict()}, self.checkpoint_path)
                logging.info(f"Checkpoint saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        return train_loss, val_loss, train_mae, val_mae, train_rmse, val_rmse

    def evaluate(self, test_loader, flat_image_paths):
        """모델 평가 및 잘못된 예측 분석"""
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        total_loss, total_mae, total_mse, total = 0.0, 0.0, 0.0, 0
        wrong = []
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                if inputs is None or labels is None or inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device).view(-1, 1)
                outputs = self.model(inputs)

                total_loss += self.criterion(outputs, labels).item()
                total_mae += torch.mean(torch.abs(outputs - labels)).item()
                total_mse += torch.mean((outputs - labels) ** 2).item()
                total += labels.size(0)
                
                errors = torch.abs(outputs-labels)
                err_rate = errors/(labels+1e-8)
                wrong_idx = (err_rate > 0.05).nonzero(as_tuple=True)[0]
                for i in wrong_idx:
                    wrong.append((flat_image_paths[idx * test_loader.batch_size + i], 
                                  labels[i].item(), outputs[i].item()))
        test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        test_mae = total_mae / len(test_loader) if len(test_loader) > 0 else 0
        test_rmse = np.sqrt(total_mse / len(test_loader)) if len(test_loader) > 0 else 0
        logging.info(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
        return test_loss, test_mae, test_rmse, wrong

def main():
    """프로그램 실행 진입점"""
    # 설정 로드
    config = load_config()
    logging.basicConfig(filename=config['log_path'], level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    print("데이터 전처리 시작")
    # 데이터 전처리 및 라벨링
    imgs = labeling(config['csv_dir_path'], config['imgs_path'])

    flat_image_paths = []
    flat_labels = []
    for lbl, paths in imgs.items():
        flat_image_paths.extend(paths)
        flat_labels.extend([lbl] * len(paths))

    # 데이터셋 생성
    try:
        dataset = CustomDataset(flat_image_paths, flat_labels, transform=TRANSFORM)
    except ValueError as e:
        logging.error(f"Dataset creation error: {e}")
        print(f"데이터셋 생성 오류: {e}")
        return
    print("데이터 전처리 완료")

    # 데이터셋 분할
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    device = get_device()
    # 모델 및 트레이너 초기화
    model = CNN().to(device)
    trainer = ModelTrainer(model, config)

    # 학습
    train_loss, val_loss, train_mae, val_mae, train_rmse, val_rmse = trainer.train(train_loader, val_loader, config['num_epochs'])

    # 평가
    test_loss, test_mae, test_rmse, wrong = trainer.evaluate(test_loader, flat_image_paths)
    print(f'최적 모델 - 테스트 손실: {test_loss:.4f}, 테스트 MAE: {test_mae:.4f}, 테스트 RMSE: {test_rmse:.4f}')
    print(f"잘못된 예측 수: {len(wrong)}")
    if wrong:
        print("잘못된 예측 상세 (최대 10개):")
        for path, true_label, pred_label in wrong[:10]:
            print(f"이미지: {path}, 실제 값: {true_label:.4f}, 예측 값: {pred_label:.4f}")

if __name__ == "__main__":
    main()