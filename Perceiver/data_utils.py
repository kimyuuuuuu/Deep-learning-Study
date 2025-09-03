"""
데이터 전처리 및 데이터셋 유틸리티
Perceiver 모델 학습을 위한 데이터 준비 함수들
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List, Optional, Union
import random


class SequenceDataset(Dataset):
    """시퀀스 데이터를 위한 커스텀 데이터셋"""
    
    def __init__(self, 
                 sequences: List[torch.Tensor], 
                 labels: List[int],
                 max_length: Optional[int] = None,
                 padding_value: float = 0.0):
        """
        Args:
            sequences: 시퀀스 데이터 리스트
            labels: 대응하는 라벨 리스트
            max_length: 최대 시퀀스 길이 (None이면 자동으로 계산)
            padding_value: 패딩에 사용할 값
        """
        self.sequences = sequences
        self.labels = labels
        self.padding_value = padding_value
        
        # 최대 길이 계산
        if max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        else:
            self.max_length = max_length
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # 패딩 또는 자르기
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        elif len(sequence) < self.max_length:
            padding = torch.full(
                (self.max_length - len(sequence), sequence.size(-1)),
                self.padding_value,
                dtype=sequence.dtype
            )
            sequence = torch.cat([sequence, padding], dim=0)
        
        # 어텐션 마스크 생성 (실제 데이터는 1, 패딩은 0)
        mask = torch.ones(len(sequence))
        original_length = len(self.sequences[idx])
        if original_length < self.max_length:
            mask[original_length:] = 0
            
        return {
            'input': sequence,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask
        }


class ImageToSequenceDataset(Dataset):
    """이미지를 시퀀스로 변환하는 데이터셋 (예: 이미지 패치를 시퀀스로)"""
    
    def __init__(self, 
                 dataset,
                 patch_size: int = 16,
                 image_size: int = 224):
        """
        Args:
            dataset: 원본 이미지 데이터셋 (CIFAR-10, ImageNet 등)
            patch_size: 패치 크기
            image_size: 입력 이미지 크기
        """
        self.dataset = dataset
        self.patch_size = patch_size
        self.image_size = image_size
        
        # 이미지를 패치로 나누는 정보 계산
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = patch_size * patch_size * 3  # RGB
        
        print(f"이미지를 {self.num_patches}개의 패치로 분할 (각 패치: {patch_size}x{patch_size})")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 이미지를 텐서로 변환 (만약 PIL이면)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # 이미지 크기 조정
        if image.size(-1) != self.image_size or image.size(-2) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 이미지를 패치로 분할
        patches = self._image_to_patches(image)
        
        return {
            'input': patches,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.ones(self.num_patches)  # 이미지는 마스킹 불필요
        }
    
    def _image_to_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        이미지를 패치로 분할
        
        Args:
            image: (C, H, W) 형태의 이미지
        Returns:
            patches: (num_patches, patch_dim) 형태의 패치들
        """
        C, H, W = image.shape
        
        # unfold를 사용하여 패치 추출
        patches = image.unfold(1, self.patch_size, self.patch_size)\
                      .unfold(2, self.patch_size, self.patch_size)
        
        # (C, num_patches_y, num_patches_x, patch_size, patch_size) 
        # -> (num_patches, patch_dim)
        patches = patches.contiguous().view(
            C, -1, self.patch_size, self.patch_size
        ).permute(1, 0, 2, 3).flatten(1)
        
        return patches


def create_synthetic_sequence_data(num_samples: int = 1000,
                                 seq_len_range: Tuple[int, int] = (50, 200),
                                 feature_dim: int = 32,
                                 num_classes: int = 5) -> Tuple[List[torch.Tensor], List[int]]:
    """
    합성 시퀀스 데이터 생성 (분류 작업용)
    
    Args:
        num_samples: 샘플 수
        seq_len_range: 시퀀스 길이 범위 (min, max)
        feature_dim: 특성 차원
        num_classes: 클래스 수
        
    Returns:
        sequences: 시퀀스 데이터 리스트
        labels: 라벨 리스트
    """
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        # 랜덤 시퀀스 길이
        seq_len = random.randint(*seq_len_range)
        
        # 클래스 결정
        label = random.randint(0, num_classes - 1)
        
        # 클래스에 따른 시퀀스 패턴 생성
        if label == 0:
            # 증가하는 패턴
            sequence = torch.randn(seq_len, feature_dim)
            sequence += torch.linspace(0, 2, seq_len).unsqueeze(1)
        elif label == 1:
            # 감소하는 패턴
            sequence = torch.randn(seq_len, feature_dim)
            sequence += torch.linspace(2, 0, seq_len).unsqueeze(1)
        elif label == 2:
            # 주기적 패턴
            t = torch.linspace(0, 4*np.pi, seq_len)
            sequence = torch.randn(seq_len, feature_dim)
            sequence += torch.sin(t).unsqueeze(1) * 2
        elif label == 3:
            # 스파이크 패턴
            sequence = torch.randn(seq_len, feature_dim) * 0.5
            spike_positions = torch.randint(0, seq_len, (seq_len//10,))
            sequence[spike_positions] += 3
        else:
            # 랜덤 노이즈
            sequence = torch.randn(seq_len, feature_dim) * 2
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels


def create_cifar10_sequence_dataset(batch_size: int = 32,
                                  patch_size: int = 8,
                                  image_size: int = 32,
                                  train: bool = True) -> DataLoader:
    """
    CIFAR-10을 패치 시퀀스로 변환한 데이터로더 생성
    
    Args:
        batch_size: 배치 크기
        patch_size: 패치 크기
        image_size: 이미지 크기 (CIFAR-10은 32x32)
        train: 학습용 여부
        
    Returns:
        DataLoader 객체
    """
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # CIFAR-10 데이터셋 로드
    cifar_dataset = datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    # 시퀀스 데이터셋으로 변환
    sequence_dataset = ImageToSequenceDataset(
        cifar_dataset, 
        patch_size=patch_size, 
        image_size=image_size
    )
    
    def collate_fn(batch):
        """배치 데이터를 적절히 정렬"""
        inputs = torch.stack([item['input'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            'input': inputs,
            'label': labels,
            'mask': masks
        }
    
    return DataLoader(
        sequence_dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn,
        num_workers=2
    )


def create_synthetic_sequence_dataloader(batch_size: int = 32,
                                       num_samples: int = 1000,
                                       seq_len_range: Tuple[int, int] = (50, 200),
                                       feature_dim: int = 32,
                                       num_classes: int = 5) -> DataLoader:
    """
    합성 시퀀스 데이터의 데이터로더 생성
    
    Args:
        batch_size: 배치 크기
        num_samples: 샘플 수
        seq_len_range: 시퀀스 길이 범위
        feature_dim: 특성 차원
        num_classes: 클래스 수
        
    Returns:
        DataLoader 객체
    """
    # 합성 데이터 생성
    sequences, labels = create_synthetic_sequence_data(
        num_samples=num_samples,
        seq_len_range=seq_len_range,
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    
    # 데이터셋 생성
    dataset = SequenceDataset(sequences, labels)
    
    def collate_fn(batch):
        """배치 데이터를 적절히 정렬"""
        inputs = torch.stack([item['input'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            'input': inputs.float(),
            'label': labels,
            'mask': masks
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def add_positional_encoding(sequences: torch.Tensor, 
                          d_model: int,
                          max_len: int = 5000) -> torch.Tensor:
    """
    시퀀스에 위치 인코딩 추가 (선택사항)
    
    Args:
        sequences: (batch_size, seq_len, feature_dim) 
        d_model: 모델 차원
        max_len: 최대 시퀀스 길이
        
    Returns:
        위치 인코딩이 추가된 시퀀스
    """
    batch_size, seq_len, _ = sequences.shape
    
    # 위치 인코딩 생성
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 배치에 맞게 확장
    pe = pe.unsqueeze(0).expand(batch_size, -1, -1)[:, :seq_len, :]
    
    return sequences + pe.to(sequences.device)


if __name__ == "__main__":
    print("=== 데이터 유틸리티 테스트 ===")
    
    # 1. 합성 시퀀스 데이터 테스트
    print("\n1. 합성 시퀀스 데이터 테스트")
    dataloader = create_synthetic_sequence_dataloader(
        batch_size=8, 
        num_samples=100,
        feature_dim=32,
        num_classes=5
    )
    
    for batch in dataloader:
        print(f"입력 크기: {batch['input'].shape}")
        print(f"라벨 크기: {batch['label'].shape}")
        print(f"마스크 크기: {batch['mask'].shape}")
        print(f"라벨 예시: {batch['label'][:5]}")
        break
    
    # 2. CIFAR-10 패치 시퀀스 데이터 테스트
    print("\n2. CIFAR-10 패치 시퀀스 데이터 테스트")
    try:
        cifar_dataloader = create_cifar10_sequence_dataset(
            batch_size=4, 
            patch_size=8
        )
        
        for batch in cifar_dataloader:
            print(f"입력 크기: {batch['input'].shape}")
            print(f"라벨 크기: {batch['label'].shape}")
            print(f"라벨 예시: {batch['label']}")
            break
    except Exception as e:
        print(f"CIFAR-10 데이터 로딩 실패: {e}")
        print("인터넷 연결을 확인하거나 데이터 다운로드를 위해 시간이 필요할 수 있습니다.")