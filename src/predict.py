import torch
import os
import pandas as pd
from PIL import Image
import argparse
from model import CNN
from utils import TRANSFORM, load_config, to_jpg, get_device
import re

def prediction(model, csv, device):
    try:
        img_dir, _ = to_jpg(pd.read_csv(csv, encoding='cp949'),'test',output_dir='..',augmentation=False)
        img_path = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]

        predicted, cycle = 0, 0
        for i in img_path:
            img = Image.open(i).convert('RGB')
            img = TRANSFORM(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                predicted += output.item()
            cycle += 1

        return predicted/cycle

    except Exception as e:
        print(f"Error in prediction: {e}")
        return


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="단일 이미지 예측")
    parser.add_argument('--csv_dir', type=str, default=None, help="csv 파일들이 담긴 폴더 경로")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="체크포인트 경로")
    args = parser.parse_args()
    
    device = get_device()
    model = CNN().to(device)

    # csv 경로가 인자로 제공되면 사용, 아니면 config.yaml의 경로 사용
    csv = args.csv_dir if args.csv_dir else config['csv_dir_path']
    ckpt = torch.load(args.checkpoint_path) if args.checkpoint_path else torch.load(config['checkpoint_path'])
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    result = list()
    for f in os.listdir(csv):
        if f.endswith('.csv'):
            csv_path = os.path.join(csv, f)
            label = prediction(model, csv_path, device)
            gt = int(re.search(r'_(\d+)%', f).group(1))/100
            err = abs(label - gt) / gt
            result.append({'filename': f, 'concentration': f'{label*100:.4f}%', 'err_rate': f"{err*100:.4f}%"})
    df = pd.DataFrame(result)
    print(df)

if __name__ == "__main__":
    main()