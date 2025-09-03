# Perceiver 모델 학습 코드

PyTorch로 구현한 Perceiver 모델의 완전한 학습 및 평가 코드입니다.

## 📋 개요

Perceiver는 DeepMind에서 제안한 범용 아키텍처로, 다양한 모달리티의 입력을 처리할 수 있습니다:
- **Cross-attention**: 입력에서 고정 크기 latent로 정보 압축
- **Self-attention**: Latent 공간에서 정보 처리
- **확장성**: 입력 크기에 관계없이 계산량 일정

## 🚀 빠른 시작

### 1. 기본 사용법

```python
from perceiver_model import create_perceiver_model

# 모델 생성
model = create_perceiver_model(
    input_dim=32,      # 입력 특성 차원
    num_classes=10,    # 분류 클래스 수
    model_size='base'  # 'small', 'base', 'large'
)

# 추론
import torch
x = torch.randn(4, 100, 32)  # (배치, 시퀀스길이, 특성차원)
output = model(x)  # (배치, 클래스수)
```

### 2. 예제 실행

```bash
# 다양한 예제 실행
python example_usage.py

# 합성 데이터로 간단한 학습
python train.py --dataset synthetic --epochs 10 --batch_size 16
```

## 📁 파일 구조

```
Perceiver/
├── perceiver_model.py    # 모델 아키텍처 구현
├── data_utils.py         # 데이터 전처리 유틸리티
├── train.py             # 학습 스크립트
├── inference.py         # 추론 및 평가 스크립트
├── example_usage.py     # 사용 예제들
└── README.md            # 이 파일
```

## 🏗️ 모델 아키텍처

### 주요 구성 요소

1. **MultiHeadAttention**: 표준 멀티헤드 어텐션
2. **CrossAttentionBlock**: 입력→latent 정보 전달
3. **SelfAttentionBlock**: Latent 공간 내 정보 처리
4. **PerceiverEncoder**: Cross + Self attention 반복
5. **PerceiverModel**: 완전한 분류 모델

### 모델 크기별 설정

| 크기   | d_model | heads | latents | layers (cross/self) |
|--------|---------|-------|---------|-------------------|
| small  | 256     | 4     | 128     | 1 / 3             |
| base   | 512     | 8     | 256     | 1 / 6             |  
| large  | 768     | 12    | 512     | 2 / 8             |

## 📊 데이터 처리

### 지원하는 데이터 타입

1. **합성 시퀀스 데이터**: 패턴별 시계열 데이터
2. **이미지→패치**: CIFAR-10을 패치 시퀀스로 변환
3. **커스텀 시퀀스**: 사용자 정의 시퀀스 데이터

### 데이터 예제

```python
from data_utils import create_synthetic_sequence_dataloader

# 합성 데이터 생성
dataloader = create_synthetic_sequence_dataloader(
    batch_size=32,
    num_samples=1000,
    seq_len_range=(50, 200),
    feature_dim=64,
    num_classes=5
)

# CIFAR-10 패치 데이터
from data_utils import create_cifar10_sequence_dataset
cifar_loader = create_cifar10_sequence_dataset(
    batch_size=32,
    patch_size=8
)
```

## 🎯 학습하기

### 기본 학습

```bash
# 합성 데이터로 학습
python train.py --dataset synthetic --model_size base --epochs 50

# CIFAR-10으로 학습  
python train.py --dataset cifar10 --model_size small --epochs 100 --lr 1e-4
```

### 고급 옵션

```bash
python train.py \
    --dataset cifar10 \
    --model_size large \
    --batch_size 64 \
    --epochs 200 \
    --lr 2e-4 \
    --config config.json
```

### 설정 파일 예제 (config.json)

```json
{
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "weight_decay": 1e-4,
    "model_size": "base",
    "dataset_type": "synthetic",
    "seq_len_range": [50, 200],
    "num_samples": 5000
}
```

## 📈 모델 평가

### 기본 평가

```bash
# 학습된 모델 평가
python inference.py --model_path checkpoints/best_model.pth

# 상세 분석 포함
python inference.py \
    --model_path checkpoints/best_model.pth \
    --analyze \
    --output_dir results/
```

### 평가 결과

평가 시 생성되는 파일들:
- `confusion_matrix.png`: 혼동 행렬
- `confidence_histogram.png`: 신뢰도 분포
- `confidence_analysis.png`: 신뢰도별 정확도 분석
- `results.json`: 수치 결과

## 🔧 커스터마이징

### 새로운 데이터셋 추가

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'input': self.data[idx],
            'label': self.labels[idx],
            'mask': torch.ones(len(self.data[idx]))
        }
```

### 모델 구조 수정

```python
# 커스텀 Perceiver 설정
model = PerceiverModel(
    input_dim=128,
    d_model=512,
    num_heads=8,
    num_cross_attention_layers=2,  # Cross-attention 레이어 수 증가
    num_self_attention_layers=8,   # Self-attention 레이어 수 증가
    num_latents=512,               # Latent 수 증가
    num_classes=1000
)
```

## 📚 주요 기능

### 1. 가변 길이 시퀀스 처리
- 자동 패딩 및 마스킹
- 효율적인 배치 처리

### 2. 멀티모달 입력 지원  
- 텍스트, 이미지, 시계열 등
- 통합된 전처리 파이프라인

### 3. 확장 가능한 아키텍처
- 모듈화된 설계
- 쉬운 커스터마이징

### 4. 완벽한 학습 파이프라인
- TensorBoard 로깅
- 체크포인트 저장/로드
- Learning rate scheduling
- Gradient clipping

## ⚡ 성능 최적화 팁

1. **배치 크기**: GPU 메모리에 맞게 조정
2. **Latent 수**: 성능과 메모리의 트레이드오프
3. **시퀀스 길이**: 너무 긴 시퀀스는 잘라내기
4. **Mixed Precision**: 대용량 모델에서 메모리 절약

```python
# Mixed precision 예제
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(inputs)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - 모델 크기 줄이기 ('small' 사용)
   - Gradient checkpointing 사용

2. **학습이 느린 경우**
   - 시퀀스 길이 줄이기
   - Latent 수 줄이기
   - 더 작은 모델 크기 사용

3. **정확도가 낮은 경우**
   - Learning rate 조정
   - 더 많은 에포크 학습
   - 모델 크기 증가

## 📖 참고 자료

- [Perceiver 논문](https://arxiv.org/abs/2103.03206)
- [Perceiver IO 논문](https://arxiv.org/abs/2107.14795)
- [PyTorch 공식 문서](https://pytorch.org/docs/)

## 🤝 기여하기

버그 리포트나 기능 개선 제안은 언제나 환영입니다!

---

## 📄 라이센스

이 코드는 교육 목적으로 제작되었습니다.