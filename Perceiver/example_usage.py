"""
Perceiver 모델 사용 예제
다양한 데이터셋과 태스크에서 Perceiver 모델을 사용하는 방법을 보여주는 예제들
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

from perceiver_model import create_perceiver_model
from data_utils import (
    create_synthetic_sequence_dataloader,
    create_cifar10_sequence_dataset,
    create_synthetic_sequence_data,
    SequenceDataset
)


def example_1_synthetic_classification():
    """예제 1: 합성 시퀀스 데이터 분류"""
    print("=" * 50)
    print("예제 1: 합성 시퀀스 데이터 분류")
    print("=" * 50)
    
    # 모델 설정
    input_dim = 32
    num_classes = 5
    batch_size = 16
    
    # 모델 생성
    model = create_perceiver_model(
        input_dim=input_dim,
        num_classes=num_classes,
        model_size='small'  # 빠른 테스트를 위해 작은 모델 사용
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 데이터 로더 생성
    train_loader = create_synthetic_sequence_dataloader(
        batch_size=batch_size,
        num_samples=200,
        seq_len_range=(30, 100),
        feature_dim=input_dim,
        num_classes=num_classes
    )
    
    # 간단한 학습 루프 (몇 번의 배치만)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("간단한 학습 시작...")
    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 3개 배치만 학습
                break
                
            inputs = batch['input']
            labels = batch['label']
            masks = batch['mask']
            
            optimizer.zero_grad()
            logits = model(inputs, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"에포크 {epoch+1}, 배치 {batch_idx+1}: 손실 = {loss.item():.4f}")
        
        print(f"에포크 {epoch+1} 평균 손실: {total_loss/3:.4f}")
    
    # 추론 예제
    print("\n추론 테스트...")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['input']
            labels = batch['label']
            masks = batch['mask']
            
            logits = model(inputs, masks)
            predictions = torch.argmax(logits, dim=-1)
            
            print(f"실제 라벨: {labels[:5].tolist()}")
            print(f"예측 라벨: {predictions[:5].tolist()}")
            break


def example_2_cifar10_patches():
    """예제 2: CIFAR-10 이미지를 패치로 분류"""
    print("\n" + "=" * 50)
    print("예제 2: CIFAR-10 패치 시퀀스 분류")
    print("=" * 50)
    
    # 설정
    patch_size = 8
    input_dim = patch_size * patch_size * 3  # RGB
    num_classes = 10
    batch_size = 8
    
    try:
        # 데이터 로더 생성
        train_loader = create_cifar10_sequence_dataset(
            batch_size=batch_size,
            patch_size=patch_size,
            train=True
        )
        
        # 모델 생성
        model = create_perceiver_model(
            input_dim=input_dim,
            num_classes=num_classes,
            model_size='small'
        )
        
        print(f"패치 크기: {patch_size}x{patch_size}")
        print(f"입력 차원: {input_dim}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 데이터 탐색
        print("\n데이터 탐색...")
        for batch in train_loader:
            inputs = batch['input']
            labels = batch['label']
            masks = batch['mask']
            
            print(f"입력 크기: {inputs.shape}")  # (batch, num_patches, patch_dim)
            print(f"라벨: {labels}")
            print(f"각 이미지의 패치 수: {inputs.shape[1]}")
            break
        
        # 간단한 순전파 테스트
        print("\n순전파 테스트...")
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            logits = model(inputs, masks)
            end_time = time.time()
            
            predictions = torch.argmax(logits, dim=-1)
            print(f"추론 시간: {end_time - start_time:.4f}초")
            print(f"예측 결과: {predictions.tolist()}")
            
    except Exception as e:
        print(f"CIFAR-10 예제 실행 실패: {e}")
        print("인터넷 연결 또는 데이터 다운로드 문제일 수 있습니다.")


def example_3_attention_visualization():
    """예제 3: 어텐션 가중치 시각화 (간단한 버전)"""
    print("\n" + "=" * 50)
    print("예제 3: 모델 구조 및 특성 분석")
    print("=" * 50)
    
    # 다양한 모델 크기 비교
    input_dim = 64
    num_classes = 10
    
    model_sizes = ['small', 'base', 'large']
    
    print("모델 크기별 파라미터 수 비교:")
    for size in model_sizes:
        model = create_perceiver_model(
            input_dim=input_dim,
            num_classes=num_classes,
            model_size=size
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{size.capitalize()} 모델: {param_count:,} 파라미터")
        
        # 메모리 사용량 테스트
        dummy_input = torch.randn(1, 50, input_dim)
        
        # 추론 시간 측정
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(dummy_input)
            end_time = time.time()
            
            print(f"  - 추론 시간: {(end_time - start_time)*1000:.2f}ms")
            print(f"  - 출력 크기: {output.shape}")


def example_4_custom_sequence_task():
    """예제 4: 커스텀 시퀀스 태스크"""
    print("\n" + "=" * 50)
    print("예제 4: 커스텀 시퀀스 태스크 (시계열 분류)")
    print("=" * 50)
    
    def create_time_series_data(num_samples=100):
        """간단한 시계열 패턴 데이터 생성"""
        sequences = []
        labels = []
        
        for _ in range(num_samples):
            seq_len = np.random.randint(40, 80)
            t = np.linspace(0, 4*np.pi, seq_len)
            
            # 3가지 패턴
            pattern_type = np.random.randint(0, 3)
            
            if pattern_type == 0:  # 사인파
                signal = np.sin(t) + np.random.normal(0, 0.1, seq_len)
                label = 0
            elif pattern_type == 1:  # 코사인파
                signal = np.cos(t) + np.random.normal(0, 0.1, seq_len)
                label = 1
            else:  # 노이즈
                signal = np.random.normal(0, 1, seq_len)
                label = 2
            
            # 다차원 특성으로 변환 (원본 신호 + 미분 + 적분)
            diff_signal = np.gradient(signal)
            cumsum_signal = np.cumsum(signal)
            
            # 정규화
            features = np.stack([
                (signal - signal.mean()) / (signal.std() + 1e-8),
                (diff_signal - diff_signal.mean()) / (diff_signal.std() + 1e-8),
                (cumsum_signal - cumsum_signal.mean()) / (cumsum_signal.std() + 1e-8)
            ], axis=1)
            
            sequences.append(torch.tensor(features, dtype=torch.float32))
            labels.append(label)
        
        return sequences, labels
    
    # 데이터 생성
    train_sequences, train_labels = create_time_series_data(200)
    test_sequences, test_labels = create_time_series_data(50)
    
    # 데이터셋 생성
    train_dataset = SequenceDataset(train_sequences, train_labels)
    test_dataset = SequenceDataset(test_sequences, test_labels)
    
    def collate_fn(batch):
        inputs = torch.stack([item['input'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        return {'input': inputs, 'label': labels, 'mask': masks}
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 모델 생성
    model = create_perceiver_model(
        input_dim=3,  # 3개 특성 (원본, 미분, 적분)
        num_classes=3,  # 3개 패턴
        model_size='small'
    )
    
    print(f"시계열 데이터 특성:")
    print(f"- 학습 샘플: {len(train_sequences)}")
    print(f"- 테스트 샘플: {len(test_sequences)}")
    print(f"- 특성 차원: 3 (원본, 미분, 적분)")
    print(f"- 클래스 수: 3 (사인파, 코사인파, 노이즈)")
    
    # 간단한 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\n시계열 분류 학습 시작...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 3배치만
                break
                
            optimizer.zero_grad()
            logits = model(batch['input'], batch['mask'])
            loss = criterion(logits, batch['label'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"에포크 {epoch+1}: 평균 손실 = {total_loss/3:.4f}")
    
    # 테스트
    print("\n테스트 결과:")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch['input'], batch['mask'])
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == batch['label']).sum().item()
            total += batch['label'].size(0)
    
    accuracy = correct / total
    print(f"테스트 정확도: {accuracy:.4f}")


def main():
    """모든 예제 실행"""
    print("Perceiver 모델 사용 예제들")
    print("=" * 60)
    
    try:
        # 예제 1: 합성 데이터 분류
        example_1_synthetic_classification()
        
        # 예제 2: CIFAR-10 패치 분류
        example_2_cifar10_patches()
        
        # 예제 3: 모델 분석
        example_3_attention_visualization()
        
        # 예제 4: 커스텀 시퀀스 태스크
        example_4_custom_sequence_task()
        
    except KeyboardInterrupt:
        print("\n예제 실행이 중단되었습니다.")
    except Exception as e:
        print(f"\n예제 실행 중 오류 발생: {e}")
    
    print("\n모든 예제 완료!")


if __name__ == "__main__":
    main()