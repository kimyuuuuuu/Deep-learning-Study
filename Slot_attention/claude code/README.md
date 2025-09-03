# Slot Attention 구현 및 학습

이 프로젝트는 DeepMind의 "Object-Centric Learning with Slot Attention" 논문을 PyTorch로 구현한 것입니다.

## 🎯 Slot Attention이란?

Slot Attention은 **감독 학습 없이** 장면을 여러 객체로 분해하는 어텐션 기반 아키텍처입니다.

### 핵심 개념:
- **슬롯(Slot)**: 각 객체를 나타내는 고정 크기 벡터
- **경쟁적 어텐션**: 슬롯들이 입력의 다른 부분을 '소유'하려고 경쟁
- **반복적 업데이트**: 여러 라운드를 거쳐 슬롯을 개선
- **순열 등변성**: 슬롯 순서에 관계없이 같은 결과

## 📁 파일 구조

```
Slot_attention/
├── slot_attention.py    # Slot Attention 모델 구현
├── dataset.py          # 데이터셋 생성 및 로딩
├── train.py           # 훈련 스크립트
├── visualize.py       # 시각화 함수들
├── requirements.txt   # 필요한 패키지들
└── README.md         # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv slot_attention_env

# 가상환경 활성화
# Windows:
slot_attention_env\Scripts\activate
# Linux/Mac:
source slot_attention_env/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 기본 훈련 실행

```bash
# 기본 설정으로 훈련 시작
python train.py

# 커스텀 설정으로 훈련
python train.py --epochs 50 --batch_size 16 --lr 1e-4 --num_slots 5
```

### 3. 데이터셋 테스트

```bash
# 데이터셋 생성 테스트
python dataset.py

# 시각화 테스트
python visualize.py
```

## 🔧 주요 매개변수

### 모델 매개변수:
- `num_slots`: 슬롯 개수 (기본값: 7)
- `slot_dim`: 슬롯 차원 (기본값: 64)
- `num_iterations`: 어텐션 반복 횟수 (기본값: 3)
- `hidden_dim`: 숨겨진 레이어 차원 (기본값: 64)

### 훈련 매개변수:
- `batch_size`: 배치 크기 (기본값: 32)
- `learning_rate`: 학습률 (기본값: 4e-4)
- `epochs`: 훈련 에폭 수 (기본값: 100)

## 📊 사용 가능한 데이터셋

### 1. Shapes Dataset (기본 제공)
```python
from dataset import create_dataloader

train_loader, val_loader = create_dataloader(
    dataset_type='shapes',
    batch_size=32,
    resolution=64,
    max_objects=6,
    min_objects=2
)
```

- **특징**: 기하학적 도형들 (원, 사각형, 삼각형)
- **장점**: 즉시 사용 가능, 빠른 실험
- **용도**: 개념 검증, 빠른 테스트

### 2. CLEVR-6 Dataset
```python
train_loader, val_loader = create_dataloader(
    dataset_type='clevr6',
    batch_size=32,
    resolution=128
)
```

- **특징**: CLEVR의 단순화 버전
- **다운로드**: [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets)

## 🎨 시각화 예시

### 1. 기본 재구성 결과
```python
from visualize import visualize_slots

# 모델 결과 시각화
recons, masks, slots, attn = model(images)
fig = visualize_slots(images, recons, masks, save_path='results.png')
```

### 2. 어텐션 진화 과정
```python
from visualize import visualize_attention_evolution

# 반복별 어텐션 변화 시각화
fig = visualize_attention_evolution(attn, save_path='attention_evolution.png')
```

### 3. 상세 슬롯 분해
```python
from visualize import visualize_slot_decomposition

# 개별 이미지의 슬롯별 분해
fig = visualize_slot_decomposition(
    original_image, reconstructed_image, masks, 
    save_path='decomposition.png'
)
```

## 📈 훈련 모니터링

### TensorBoard 사용
```bash
# 훈련 중 TensorBoard 실행
tensorboard --logdir logs

# 브라우저에서 http://localhost:6006 접속
```

### 체크포인트 관리
```python
# 훈련 재개
python train.py --resume logs/checkpoints/best_model.pth

# 특정 에폭에서 재개
python train.py --resume logs/checkpoints/checkpoint_epoch_50.pth
```

## 🔍 실험 가이드

### 1. 하이퍼파라미터 조정
```bash
# 슬롯 개수 실험
python train.py --num_slots 5   # 간단한 장면
python train.py --num_slots 10  # 복잡한 장면

# 학습률 실험
python train.py --lr 1e-4   # 안정적 학습
python train.py --lr 1e-3   # 빠른 학습

# 해상도 실험
python train.py --resolution 32   # 빠른 실험
python train.py --resolution 128  # 고해상도
```

### 2. 데이터셋별 권장 설정

**Shapes Dataset:**
```bash
python train.py --dataset shapes --num_slots 7 --resolution 64 --epochs 100
```

**CLEVR-6 Dataset:**
```bash
python train.py --dataset clevr6 --num_slots 7 --resolution 128 --epochs 200
```

## 🎓 학습 팁

### 1. 좋은 결과를 얻기 위한 팁:
- **슬롯 개수**: 예상 객체 수보다 약간 많게 설정
- **학습률**: 4e-4에서 시작해서 필요시 조정
- **배치 크기**: GPU 메모리에 맞게 조정 (16-64 권장)
- **에폭 수**: 손실이 수렴할 때까지 (보통 100-300)

### 2. 문제 해결:
- **슬롯이 객체를 제대로 분리하지 못할 때**: `num_iterations` 증가
- **훈련이 불안정할 때**: 학습률 감소, 그래디언트 클리핑 적용
- **메모리 부족**: 배치 크기 또는 해상도 감소

### 3. 평가 방법:
- **재구성 품질**: MSE 손실 확인
- **객체 분리**: 마스크 시각화로 확인
- **일관성**: 같은 객체가 같은 슬롯에 할당되는지 확인

## 📚 참고 자료

### 논문:
- [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055) (Locatello et al., 2020)

### 관련 자료:
- [Official Implementation](https://github.com/google-research/google-research/tree/master/slot_attention)
- [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets)

## 🐛 문제 신고

버그나 문제가 있다면 GitHub 이슈를 통해 신고해주세요.

## 📄 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다. 원본 논문의 라이선스를 확인해주세요.

---

**즐거운 Slot Attention 학습 되세요! 🚀**