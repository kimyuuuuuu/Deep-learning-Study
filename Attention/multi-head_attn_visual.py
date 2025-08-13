import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import accuracy_score
import time

from module.MultiHead_Attention import MultiHeadAttention

# 1. 기본 검증: 차원 확인
def test_dimensions():
    """MultiHeadAttention의 입출력 차원이 올바른지 검증"""
    print("=== 1. 차원 검증 ===")
    
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # 테스트 데이터
    x = torch.randn(batch_size, seq_len, d_model)
    
    # MultiHeadAttention 생성
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attn_weights = mha(x, x, x)
    
    print(f"입력 차원: {x.shape}")
    print(f"출력 차원: {output.shape}")
    print(f"Attention weights 차원: {attn_weights.shape}")
    
    # 검증
    assert output.shape == x.shape, f"출력 차원 불일치: {output.shape} != {x.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
           f"Attention weights 차원 불일치: {attn_weights.shape}"
    
    print("✅ 차원 검증 통과!")
    return mha, x, attn_weights

# 2. Head별 패턴 시각화
def visualize_head_patterns(mha, x, attn_weights, sentence_tokens=None):
    """각 head가 학습하는 attention pattern 시각화"""
    print("\n=== 2. Head별 패턴 시각화 ===")
    
    if sentence_tokens is None:
        sentence_tokens = [f"token_{i}" for i in range(x.size(1))]
    
    num_heads = attn_weights.size(1)
    
    # Head별 시각화
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('각 Head별 Attention Patterns', fontsize=16)
    
    for head in range(num_heads):
        row = head // 4
        col = head % 4
        
        # 첫 번째 배치의 attention matrix
        attn_matrix = attn_weights[0, head].detach().numpy()
        
        sns.heatmap(attn_matrix,
                   xticklabels=sentence_tokens,
                   yticklabels=sentence_tokens,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues',
                   ax=axes[row, col])
        
        axes[row, col].set_title(f'Head {head + 1}')
        axes[row, col].set_xlabel('Key (attending to)')
        axes[row, col].set_ylabel('Query (attending from)')
    
    plt.tight_layout()
    plt.show()
    
    # Head별 특성 분석
    analyze_head_characteristics(attn_weights, sentence_tokens)

def analyze_head_characteristics(attn_weights, tokens):
    """각 head의 특성 분석"""
    print("\n=== Head별 특성 분석 ===")
    
    num_heads = attn_weights.size(1)
    seq_len = attn_weights.size(2)
    
    for head in range(num_heads):
        attn_matrix = attn_weights[0, head].detach().numpy()
        
        # Self-attention 비율
        self_attn = np.diag(attn_matrix).mean()
        
        # Attention 분산도 (entropy)
        entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-8), axis=-1).mean()
        
        # 최대 attention 평균
        max_attn = attn_matrix.max(axis=-1).mean()
        
        print(f"Head {head + 1}:")
        print(f"  Self-attention 평균: {self_attn:.3f}")
        print(f"  Attention 엔트로피: {entropy:.3f} (낮을수록 집중적)")
        print(f"  최대 attention 평균: {max_attn:.3f}")
        print()

# 3. Head 수에 따른 성능 비교
def compare_head_numbers():
    """Head 수에 따른 성능 변화 관찰"""
    print("=== 3. Head 수에 따른 성능 비교 ===")
    
    d_model = 512
    seq_len = 20
    batch_size = 32
    num_epochs = 10
    
    # 테스트할 head 수들
    head_numbers = [1, 2, 4, 8, 16]
    results = []
    
    for num_heads in head_numbers:
        print(f"\n🔹 Testing {num_heads} heads...")
        
        # 간단한 분류 태스크 설정
        model = SimpleClassifier(d_model, num_heads, num_classes=2)
        
        # 더미 데이터 생성 (실제로는 실제 데이터 사용)
        train_data = create_dummy_classification_data(batch_size * 10, seq_len, d_model)
        
        # 훈련 및 평가
        accuracy, training_time = train_and_evaluate(model, train_data, num_epochs)
        
        results.append({
            'num_heads': num_heads,
            'accuracy': accuracy,
            'training_time': training_time,
            'params': count_parameters(model)
        })
        
        print(f"  정확도: {accuracy:.3f}")
        print(f"  훈련 시간: {training_time:.2f}s")
        print(f"  파라미터 수: {count_parameters(model):,}")
    
    # 결과 시각화
    plot_head_comparison(results)
    return results

def create_dummy_classification_data(num_samples, seq_len, d_model):
    """더미 분류 데이터 생성"""
    X = torch.randn(num_samples, seq_len, d_model)
    # 간단한 패턴: 첫 번째 토큰의 합이 양수면 클래스 1, 음수면 클래스 0
    y = (X[:, 0, :].sum(dim=1) > 0).long()
    return X, y

class SimpleClassifier(nn.Module):
    """MultiHeadAttention을 사용하는 간단한 분류기"""
    def __init__(self, d_model, num_heads, num_classes):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Residual connection + layer norm
        x = self.norm(x + attn_output)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        output = self.classifier(x)
        
        return output, attn_weights

def train_and_evaluate(model, data, num_epochs):
    """모델 훈련 및 평가"""
    X, y = data
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    model.train()
    for epoch in range(num_epochs):
        # 미니배치 처리 (간단히 전체 데이터 사용)
        optimizer.zero_grad()
        
        outputs, _ = model(X)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
    
    training_time = time.time() - start_time
    
    # 평가
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = accuracy_score(y.numpy(), predicted.numpy())
    
    return accuracy, training_time

def count_parameters(model):
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_head_comparison(results):
    """Head 수 비교 결과 시각화"""
    head_nums = [r['num_heads'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['training_time'] for r in results]
    params = [r['params'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 정확도
    axes[0].plot(head_nums, accuracies, 'o-', color='blue')
    axes[0].set_xlabel('Number of Heads')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('정확도 vs Head 수')
    axes[0].grid(True)
    
    # 훈련 시간
    axes[1].plot(head_nums, times, 'o-', color='red')
    axes[1].set_xlabel('Number of Heads')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('훈련 시간 vs Head 수')
    axes[1].grid(True)
    
    # 파라미터 수
    axes[2].plot(head_nums, params, 'o-', color='green')
    axes[2].set_xlabel('Number of Heads')
    axes[2].set_ylabel('Parameters')
    axes[2].set_title('파라미터 수 vs Head 수')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# 4. 실제 문장으로 테스트
def test_with_real_sentence():
    """실제 문장으로 MultiHeadAttention 테스트"""
    print("=== 4. 실제 문장 테스트 ===")
    
    # 간단한 토크나이저
    vocab = {'<PAD>': 0, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6}
    sentence = "the cat sat on the mat"
    tokens = [vocab.get(word, 0) for word in sentence.split()]
    token_names = sentence.split()
    
    print(f"문장: {sentence}")
    print(f"토큰들: {token_names}")
    
    # Embedding
    d_model = 128
    num_heads = 4
    embedding = nn.Embedding(len(vocab), d_model)
    
    # 토큰을 벡터로 변환
    token_tensor = torch.tensor(tokens).unsqueeze(0)
    x = embedding(token_tensor)
    
    # MultiHeadAttention 적용
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(x, x, x)
    
    # 시각화
    visualize_head_patterns(mha, x, attn_weights, token_names)
    
    return output, attn_weights

# 5. Attention Head 다양성 분석
def analyze_head_diversity(attn_weights):
    """Head 간의 다양성 분석"""
    print("=== 5. Head 다양성 분석 ===")
    
    num_heads = attn_weights.size(1)
    
    # Head 간 correlation 계산
    correlations = []
    for i in range(num_heads):
        for j in range(i+1, num_heads):
            attn_i = attn_weights[0, i].flatten()
            attn_j = attn_weights[0, j].flatten()
            
            corr = torch.corrcoef(torch.stack([attn_i, attn_j]))[0, 1].item()
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    print(f"Head 간 평균 상관관계: {avg_correlation:.3f}")
    
    if avg_correlation < 0.5:
        print("✅ Head들이 다양한 패턴을 학습하고 있습니다!")
    else:
        print("⚠️  Head들이 유사한 패턴을 학습하고 있습니다.")
    
    # 상관관계 히트맵
    plt.figure(figsize=(8, 6))
    corr_matrix = np.zeros((num_heads, num_heads))
    
    idx = 0
    for i in range(num_heads):
        for j in range(i+1, num_heads):
            corr_matrix[i, j] = correlations[idx]
            corr_matrix[j, i] = correlations[idx]
            idx += 1
    
    # 대각선은 1로 설정
    np.fill_diagonal(corr_matrix, 1.0)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True)
    plt.title('Head 간 Attention Pattern 상관관계')
    plt.xlabel('Head')
    plt.ylabel('Head')
    plt.show()

# 전체 실행 함수
def run_multihead_validation():
    """전체 검증 프로세스 실행"""
    print("🎯 MultiHeadAttention 검증 시작!")
    print("=" * 50)
    
    # 1. 기본 검증
    mha, x, attn_weights = test_dimensions()
    
    # 2. 실제 문장 테스트
    output, attn_weights = test_with_real_sentence()
    
    # 3. Head 다양성 분석
    analyze_head_diversity(attn_weights)
    
    # 4. Head 수 비교 (시간이 오래 걸릴 수 있음)
    print("\n⏰ Head 수 비교는 시간이 오래 걸립니다. 실행하시겠습니까? (y/n)")
    choice = input().lower()
    if choice == 'y':
        results = compare_head_numbers()
        return results
    
    print("\n🎉 검증 완료!")

if __name__ == "__main__":
    run_multihead_validation()