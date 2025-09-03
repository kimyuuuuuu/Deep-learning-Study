import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import random
import os
import urllib.request
import zipfile
import tarfile
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ShapesDataset(Dataset):
    """
    간단한 기하학적 도형 데이터셋 생성기
    Slot Attention 학습을 위한 multi-object 씬 생성
    
    각 이미지는 여러 개의 간단한 도형(원, 사각형, 삼각형)을 포함하며,
    각 도형은 다른 색상과 크기를 가집니다.
    """
    
    def __init__(self, num_samples=10000, resolution=64, max_objects=6, min_objects=2):
        """
        Args:
            num_samples: 생성할 샘플 수
            resolution: 이미지 해상도 (정사각형)
            max_objects: 이미지당 최대 객체 수
            min_objects: 이미지당 최소 객체 수
        """
        self.num_samples = num_samples
        self.resolution = resolution
        self.max_objects = max_objects
        self.min_objects = min_objects
        
        # 사용할 색상들 (RGB)
        self.colors = [
            (255, 0, 0),    # 빨강
            (0, 255, 0),    # 초록  
            (0, 0, 255),    # 파랑
            (255, 255, 0),  # 노랑
            (255, 0, 255),  # 자홍
            (0, 255, 255),  # 청록
            (255, 128, 0),  # 주황
            (128, 0, 255),  # 보라
        ]
        
        # 도형 크기 범위 (반지름 또는 한 변의 길이)
        self.min_size = resolution // 10
        self.max_size = resolution // 4
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        하나의 multi-object 이미지 생성
        """
        # 빈 캔버스 생성 (검은 배경)
        image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 이미지에 넣을 객체 수 결정
        num_objects = random.randint(self.min_objects, self.max_objects)
        
        # 각 객체 생성
        for _ in range(num_objects):
            # 색상 선택
            color = random.choice(self.colors)
            
            # 크기 선택
            size = random.randint(self.min_size, self.max_size)
            
            # 위치 선택 (경계에서 충분히 떨어진 곳)
            margin = size + 5
            x = random.randint(margin, self.resolution - margin)
            y = random.randint(margin, self.resolution - margin)
            
            # 도형 타입 선택
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            
            if shape_type == 'circle':
                # 원 그리기
                draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
                
            elif shape_type == 'rectangle':
                # 사각형 그리기
                draw.rectangle([x-size, y-size, x+size, y+size], fill=color)
                
            elif shape_type == 'triangle':
                # 삼각형 그리기
                points = [
                    (x, y-size),           # 위
                    (x-size, y+size//2),   # 왼쪽 아래
                    (x+size, y+size//2)    # 오른쪽 아래
                ]
                draw.polygon(points, fill=color)
        
        # PIL Image를 tensor로 변환
        image_tensor = transforms.ToTensor()(image)
        
        return image_tensor


class CLEVR6Dataset(Dataset):
    """
    CLEVR-6 데이터셋 로더
    CLEVR-6은 CLEVR의 단순화된 버전으로 최대 6개의 객체를 포함
    """
    
    def __init__(self, data_dir='./clevr6_data', split='train', download=True, resolution=128):
        """
        Args:
            data_dir: 데이터를 저장할 디렉토리
            split: 'train' 또는 'val'
            download: 자동으로 데이터셋을 다운로드할지 여부
            resolution: 이미지 크기 조정
        """
        self.data_dir = data_dir
        self.split = split
        self.resolution = resolution
        
        # 변환: 이미지 크기 조정 및 정규화
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])
        
        if download:
            self._download_clevr6()
            
        # 이미지 파일 경로들 수집
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            self.image_paths = [
                os.path.join(split_dir, f) for f in os.listdir(split_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
        else:
            print(f"경고: {split_dir} 디렉토리를 찾을 수 없습니다. ShapesDataset을 대신 사용하세요.")
            self.image_paths = []
    
    def _download_clevr6(self):
        """
        CLEVR-6 데이터셋 다운로드 (실제로는 더미 구현)
        실제 사용 시에는 공식 CLEVR-6 데이터셋 다운로드 링크를 사용해야 합니다.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("CLEVR-6 데이터셋을 다운로드하려면 수동으로 다음 사이트에서 다운로드해주세요:")
            print("https://github.com/deepmind/multi_object_datasets")
            print(f"다운로드 후 {self.data_dir} 폴더에 압축을 해제해주세요.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if len(self.image_paths) == 0:
            # CLEVR-6 데이터가 없으면 임시로 shapes 데이터 사용
            shapes_dataset = ShapesDataset(resolution=self.resolution)
            return shapes_dataset[idx % 1000]
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def create_dataloader(dataset_type='shapes', batch_size=32, resolution=64, 
                     num_workers=4, **kwargs):
    """
    데이터로더 생성 함수
    
    Args:
        dataset_type: 'shapes' 또는 'clevr6'
        batch_size: 배치 크기
        resolution: 이미지 해상도
        num_workers: 데이터 로딩 워커 수
        **kwargs: 데이터셋별 추가 파라미터
        
    Returns:
        train_loader, val_loader: 훈련 및 검증 데이터로더
    """
    
    if dataset_type == 'shapes':
        # Shapes 데이터셋 사용
        train_dataset = ShapesDataset(
            num_samples=kwargs.get('train_samples', 10000),
            resolution=resolution,
            max_objects=kwargs.get('max_objects', 6),
            min_objects=kwargs.get('min_objects', 2)
        )
        
        val_dataset = ShapesDataset(
            num_samples=kwargs.get('val_samples', 1000),
            resolution=resolution,
            max_objects=kwargs.get('max_objects', 6),
            min_objects=kwargs.get('min_objects', 2)
        )
        
    elif dataset_type == 'clevr6':
        # CLEVR-6 데이터셋 사용
        train_dataset = CLEVR6Dataset(
            split='train',
            resolution=resolution,
            download=True
        )
        
        val_dataset = CLEVR6Dataset(
            split='val', 
            resolution=resolution,
            download=True
        )
        
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_type}")
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def download_sample_datasets():
    """
    샘플 데이터셋 다운로드 함수
    실제 연구에서 사용되는 데이터셋들에 대한 정보 제공
    """
    print("=== Slot Attention 학습을 위한 데이터셋 정보 ===")
    print()
    
    print("1. Multi-Object Datasets (DeepMind)")
    print("   - CLEVR-6: 최대 6개 객체가 있는 CLEVR 이미지")
    print("   - Tetrominoes: 테트리스 블록 조합")
    print("   - 다운로드: https://github.com/deepmind/multi_object_datasets")
    print()
    
    print("2. MOVi (Multi-Object Video)")  
    print("   - MOVi-A, MOVi-B, MOVi-C: 비디오 객체 분할")
    print("   - 다운로드: https://github.com/google-research/kubric/tree/main/challenges/movi")
    print()
    
    print("3. 간단한 Shapes 데이터셋 (코드에 포함됨)")
    print("   - 기하학적 도형들로 구성된 합성 데이터")
    print("   - 즉시 사용 가능 - 별도 다운로드 불필요")
    print()
    
    print("현재 코드에서는 Shapes 데이터셋이 기본으로 제공됩니다.")
    print("실험을 위해 더 복잡한 데이터셋이 필요하다면 위 링크를 참조하세요.")


if __name__ == "__main__":
    # 데이터셋 테스트
    print("=== Slot Attention 데이터셋 테스트 ===")
    
    # Shapes 데이터셋 테스트
    print("Shapes 데이터셋 생성 중...")
    train_loader, val_loader = create_dataloader(
        dataset_type='shapes',
        batch_size=4,
        resolution=64,
        train_samples=100,
        val_samples=20
    )
    
    # 샘플 확인
    sample_batch = next(iter(train_loader))
    print(f"배치 모양: {sample_batch.shape}")
    print(f"데이터 타입: {sample_batch.dtype}")
    print(f"값 범위: [{sample_batch.min().item():.3f}, {sample_batch.max().item():.3f}]")
    
    # 다운로드 정보 출력
    download_sample_datasets()