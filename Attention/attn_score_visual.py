import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from module.Scaled_dot_product_Attention import ScaleDotProductAttention

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SimpleTokenizer:
    def __init__(self):
        # 간단한 vocabulary
        self.vocab = {
            '<PAD>': 0, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5,
            'dog': 6, 'ran': 7, 'to': 8, 'park': 9, 'and': 10, 'played': 11
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, sentence):
        tokens = sentence.lower().split()
        return [self.vocab.get(token, 0) for token in tokens]
    
    def decode(self, ids):
        return [self.id_to_token.get(id, '<UNK>') for id in ids]

def create_simple_embeddings(vocab_size=20, d_model=64):
    """간단한 word embedding 생성 (실제로는 학습된 것 사용)"""
    # 재현 가능한 결과를 위해 seed 설정
    torch.manual_seed(42)
    embedding = nn.Embedding(vocab_size, d_model)
    return embedding

def visualize_attention_simple():
    """가장 간단한 attention 시각화"""
    
    # 1. 데이터 준비
    tokenizer = SimpleTokenizer()
    sentence = "the cat sat on the mat"
    tokens = tokenizer.tokenize(sentence)
    token_names = tokenizer.decode(tokens)
    
    print(f"문장: {sentence}")
    print(f"토큰들: {token_names}")
    print(f"토큰 IDs: {tokens}")
    
    # 2. Embedding 생성
    d_model = 64
    embedding = create_simple_embeddings(d_model=d_model)
    
    # 토큰을 embedding으로 변환
    token_tensor = torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]
    embeddings = embedding(token_tensor)  # [1, seq_len, d_model]
    
    # 3. Multi-head attention을 위한 차원 변환
    num_heads = 4
    d_k = d_model // num_heads
    seq_len = len(tokens)
    
    # Q, K, V 생성 (간단히 같은 embedding 사용)
    # 실제로는 서로 다른 linear projection 사용
    q = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)  # [1, num_heads, seq_len, d_k]
    k = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    v = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 4. Attention 계산
    attention_layer = ScaleDotProductAttention()
    output, attention_weights = attention_layer(q, k, v)
    
    # 5. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Attention Weights for: "{sentence}"', fontsize=16)
    
    # 각 head별로 시각화
    for head in range(num_heads):
        row = head // 2
        col = head % 2
        
        # attention weights 추출 [seq_len, seq_len]
        attn_matrix = attention_weights[0, head].detach().numpy()
        
        # Heatmap 그리기
        sns.heatmap(attn_matrix, 
                   xticklabels=token_names,
                   yticklabels=token_names,
                   annot=True,
                   fmt='.3f',
                   cmap='Blues',
                   ax=axes[row, col])
        
        axes[row, col].set_title(f'Head {head + 1}')
        axes[row, col].set_xlabel('Key (attending to)')
        axes[row, col].set_ylabel('Query (attending from)')
    
    plt.tight_layout()
    plt.show()
    
    # 6. 전체 평균 attention도 보기
    plt.figure(figsize=(8, 6))
    avg_attention = attention_weights.mean(dim=1)[0].detach().numpy()  # 모든 head 평균
    
    sns.heatmap(avg_attention,
               xticklabels=token_names,
               yticklabels=token_names,
               annot=True,
               fmt='.3f',
               cmap='Reds')
    
    plt.title('Average Attention Across All Heads')
    plt.xlabel('Key (attending to)')
    plt.ylabel('Query (attending from)')
    plt.show()
    
    return attention_weights, token_names

def analyze_attention_patterns(attention_weights, tokens):
    """Attention 패턴 분석"""
    
    print("\n=== Attention 패턴 분석 ===")
    
    # 각 단어가 가장 많이 주목하는 단어 찾기
    avg_attention = attention_weights.mean(dim=1)[0].detach().numpy()
    
    for i, token in enumerate(tokens):
        # 자기 자신 제외하고 가장 높은 attention을 받는 단어
        attention_scores = avg_attention[i].copy()
        attention_scores[i] = 0  # 자기 자신 제외
        
        max_idx = np.argmax(attention_scores)
        max_score = attention_scores[max_idx]
        
        print(f"'{token}' -> '{tokens[max_idx]}' (attention: {max_score:.3f})")
    
    print(f"\n각 위치의 attention 합계 (모두 1.0이어야 함):")
    for i, token in enumerate(tokens):
        attention_sum = avg_attention[i].sum()
        print(f"'{token}': {attention_sum:.3f}")

def compare_different_sentences():
    """여러 문장의 attention 패턴 비교"""
    
    sentences = [
        "the cat sat on the mat",
        "the dog ran to the park", 
        "the cat and dog played"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    tokenizer = SimpleTokenizer()
    embedding = create_simple_embeddings()
    attention_layer = ScaleDotProductAttention()
    
    for idx, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(sentence)
        token_names = tokenizer.decode(tokens)
        
        # Embedding 및 attention 계산
        token_tensor = torch.tensor(tokens).unsqueeze(0)
        embeddings = embedding(token_tensor)
        
        seq_len = len(tokens)
        d_model = 64
        num_heads = 4
        d_k = d_model // num_heads
        
        q = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)
        k = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)  
        v = embeddings.view(1, seq_len, num_heads, d_k).transpose(1, 2)
        
        _, attention_weights = attention_layer(q, k, v)
        avg_attention = attention_weights.mean(dim=1)[0].detach().numpy()
        
        # 시각화
        sns.heatmap(avg_attention,
                   xticklabels=token_names,
                   yticklabels=token_names,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues',
                   ax=axes[idx])
        
        axes[idx].set_title(f'"{sentence}"')
        axes[idx].set_xlabel('Key')
        axes[idx].set_ylabel('Query')
    
    plt.tight_layout()
    plt.show()

# 실행 함수들
if __name__ == "__main__":
    print("🎯 Attention Score 시각화 실습")
    print("=" * 50)
    
    # 1. 기본 시각화
    print("1. 기본 attention 시각화")
    attention_weights, tokens = visualize_attention_simple()
    
    # 2. 패턴 분석  
    analyze_attention_patterns(attention_weights, tokens)
    
    # 3. 여러 문장 비교
    print("\n2. 여러 문장 attention 패턴 비교")
    compare_different_sentences()