"""
Perceiver 모델 구현
DeepMind의 Perceiver 아키텍처를 PyTorch로 구현한 버전
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 모듈"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value 변환을 위한 선형 레이어
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)  
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) 또는 None
        """
        batch_size = query.size(0)
        
        # Q, K, V 계산 및 멀티헤드로 reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask가 0인 곳을 매우 작은 값으로 설정
            # Cross-attention의 경우: mask는 (batch_size, seq_len_k) 형태
            # Self-attention의 경우: mask는 (batch_size, seq_len_q, seq_len_k) 형태일 수 있음
            
            if mask.dim() == 2:  # Cross-attention: (batch_size, seq_len_k)
                # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k)
                mask = mask.unsqueeze(1).unsqueeze(1)
                # (batch_size, 1, 1, seq_len_k) -> (batch_size, num_heads, seq_len_q, seq_len_k)
                mask = mask.expand(-1, self.num_heads, scores.size(2), -1)
            elif mask.dim() == 3:  # Self-attention: (batch_size, seq_len_q, seq_len_k)
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, num_heads, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 어텐션 가중치 저장 (시각화용)
        self.last_attention_weights = attention_weights.detach()
        
        # 어텐션 적용
        context = torch.matmul(attention_weights, V)
        
        # 멀티헤드 결합
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """Feed Forward 네트워크"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class CrossAttentionBlock(nn.Module):
    """Cross-Attention 블록 - 입력에서 latent로 정보를 전달"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent: torch.Tensor, inputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latent: (batch_size, num_latents, d_model) - 학습 가능한 쿼리
            inputs: (batch_size, input_seq_len, d_model) - 입력 데이터
            mask: 마스킹을 위한 텐서
        """
        # Cross-attention: latent가 query, inputs가 key/value
        attended = self.cross_attention(latent, inputs, inputs, mask)
        latent = self.norm1(latent + self.dropout(attended))
        
        # Feed Forward
        ff_output = self.feed_forward(latent)
        latent = self.norm2(latent + self.dropout(ff_output))
        
        return latent


class SelfAttentionBlock(nn.Module):
    """Self-Attention 블록 - latent 공간에서의 self-attention"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latent: (batch_size, num_latents, d_model)
            mask: 마스킹을 위한 텐서
        """
        # Self-attention
        attended = self.self_attention(latent, latent, latent, mask)
        latent = self.norm1(latent + self.dropout(attended))
        
        # Feed Forward
        ff_output = self.feed_forward(latent)
        latent = self.norm2(latent + self.dropout(ff_output))
        
        return latent


class PerceiverEncoder(nn.Module):
    """Perceiver 인코더 - Cross-attention과 Self-attention을 반복"""
    
    def __init__(self, 
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_cross_attention_layers: int = 1,
                 num_self_attention_layers: int = 6,
                 num_latents: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_latents = num_latents
        self.d_model = d_model
        
        # 학습 가능한 latent 쿼리들 초기화
        self.latent_embeddings = nn.Parameter(
            torch.randn(num_latents, d_model) * 0.02
        )
        
        # Cross-attention 레이어들
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_cross_attention_layers)
        ])
        
        # Self-attention 레이어들
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_self_attention_layers)
        ])
        
    def forward(self, inputs: torch.Tensor, 
                input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, input_seq_len, d_model) - 인코딩된 입력
            input_mask: 입력 마스킹을 위한 텐서
        Returns:
            latent: (batch_size, num_latents, d_model) - 압축된 표현
        """
        batch_size = inputs.size(0)
        
        # latent를 배치 크기에 맞게 복사
        latent = self.latent_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone()
        
        # Cross-attention으로 입력 정보를 latent로 전달
        for cross_layer in self.cross_attention_layers:
            latent = cross_layer(latent, inputs, input_mask)
        
        # Self-attention으로 latent 공간에서 정보 처리
        for self_layer in self.self_attention_layers:
            latent = self_layer(latent)
        
        return latent


class InputProjection(nn.Module):
    """입력 데이터를 모델 차원으로 투영"""
    
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - 원본 입력
        Returns:
            (batch_size, seq_len, d_model) - 투영된 입력
        """
        return self.layer_norm(self.projection(x))


class PerceiverModel(nn.Module):
    """완전한 Perceiver 모델"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_cross_attention_layers: int = 1,
                 num_self_attention_layers: int = 6,
                 num_latents: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 입력 특성의 차원
            d_model: 모델의 hidden 차원
            num_heads: 어텐션 헤드 수
            d_ff: Feed Forward 차원
            num_cross_attention_layers: Cross-attention 레이어 수
            num_self_attention_layers: Self-attention 레이어 수
            num_latents: Latent 벡터의 수
            num_classes: 분류할 클래스 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # 입력 투영 레이어
        self.input_projection = InputProjection(input_dim, d_model)
        
        # Perceiver 인코더
        self.encoder = PerceiverEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_cross_attention_layers=num_cross_attention_layers,
            num_self_attention_layers=num_self_attention_layers,
            num_latents=num_latents,
            dropout=dropout
        )
        
        # 분류를 위한 출력 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes)
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, 
                input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - 입력 데이터
            input_mask: 마스킹을 위한 텐서
        Returns:
            (batch_size, num_classes) - 클래스 예측 확률
        """
        # 입력을 모델 차원으로 투영
        x = self.input_projection(x)
        
        # Perceiver 인코더로 압축된 표현 생성
        latent = self.encoder(x, input_mask)
        
        # Global average pooling으로 단일 벡터로 압축
        # (batch_size, num_latents, d_model) -> (batch_size, d_model)
        pooled = self.pool(latent.transpose(1, 2)).squeeze(-1)
        
        # 분류 예측
        logits = self.classifier(pooled)
        
        return logits


def create_perceiver_model(input_dim: int, num_classes: int = 10, 
                          model_size: str = 'base', num_latents: int = None) -> PerceiverModel:
    """
    Perceiver 모델을 생성하는 헬퍼 함수
    
    Args:
        input_dim: 입력 특성의 차원
        num_classes: 분류할 클래스 수
        model_size: 모델 크기 ('small', 'base', 'large')
    """
    if model_size == 'small':
        config = {
            'd_model': 256,
            'num_heads': 4,
            'd_ff': 1024,
            'num_cross_attention_layers': 1,
            'num_self_attention_layers': 3,
            'num_latents': 128
        }
    elif model_size == 'base':
        config = {
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'num_cross_attention_layers': 1,
            'num_self_attention_layers': 6,
            'num_latents': 256
        }
    else:  # large
        config = {
            'd_model': 768,
            'num_heads': 12,
            'd_ff': 3072,
            'num_cross_attention_layers': 2,
            'num_self_attention_layers': 8,
            'num_latents': 512
        }
    
    # num_latents가 지정된 경우 덮어쓰기
    if num_latents is not None:
        config['num_latents'] = num_latents
    
    return PerceiverModel(
        input_dim=input_dim,
        num_classes=num_classes,
        **config
    )


if __name__ == "__main__":
    # 간단한 테스트
    batch_size, seq_len, input_dim = 4, 100, 32
    num_classes = 10
    
    # 모델 생성
    model = create_perceiver_model(input_dim, num_classes, 'base')
    
    # 더미 입력 데이터
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 순전파
    with torch.no_grad():
        output = model(x)
        print(f"입력 크기: {x.shape}")
        print(f"출력 크기: {output.shape}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")