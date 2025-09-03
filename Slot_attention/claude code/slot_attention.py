import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlotAttention(nn.Module):
    """
    Slot Attention 모듈
    
    논문: "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)
    
    주요 개념:
    - 슬롯(slot): 객체를 나타내는 고정 크기 벡터
    - 경쟁적 어텐션: 각 슬롯이 입력의 다른 부분을 '소유'하려고 경쟁
    - 반복적 업데이트: 여러 번의 어텐션 라운드를 통해 슬롯을 개선
    """
    
    def __init__(self, num_slots, dim_slot, dim_hidden=128, num_iterations=3):
        """
        Args:
            num_slots: 슬롯의 개수 (발견하려는 객체의 최대 개수)
            dim_slot: 각 슬롯의 차원
            dim_hidden: 숨겨진 레이어의 차원
            num_iterations: 어텐션 업데이트 반복 횟수
        """
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.num_iterations = num_iterations
        
        # 슬롯 초기화를 위한 파라미터들
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim_slot))  # 평균
        self.slot_sigma = nn.Parameter(torch.randn(1, 1, dim_slot))  # 표준편차
        
        # 어텐션 메커니즘을 위한 선형 변환들
        self.to_q = nn.Linear(dim_slot, dim_slot, bias=False)  # query 생성
        self.to_k = nn.Linear(dim_slot, dim_slot, bias=False)  # key 생성
        self.to_v = nn.Linear(dim_slot, dim_slot, bias=False)  # value 생성
        
        # GRU 기반 슬롯 업데이트
        self.gru = nn.GRUCell(dim_slot, dim_slot)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim_slot, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_slot)
        )
        
        # 정규화 레이어들
        self.norm_input = nn.LayerNorm(dim_slot)
        self.norm_slot = nn.LayerNorm(dim_slot)
        self.norm_mlp = nn.LayerNorm(dim_slot)
        
    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, num_inputs, input_dim) - 인코딩된 입력 특징들
            
        Returns:
            slots: (batch_size, num_slots, dim_slot) - 학습된 슬롯들
            attn_weights: (batch_size, num_iterations, num_inputs, num_slots) - 어텐션 가중치들
        """
        batch_size, num_inputs, input_dim = inputs.shape
        
        # 입력 정규화
        inputs = self.norm_input(inputs)
        
        # 슬롯 초기화: 가우시안 분포에서 샘플링
        slots = self.slot_mu + self.slot_sigma * torch.randn(
            batch_size, self.num_slots, self.dim_slot, 
            device=inputs.device, dtype=inputs.dtype
        )
        
        attn_weights_all = []
        
        # 반복적 어텐션 업데이트
        for iteration in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slot(slots)
            
            # 어텐션 계산: Q(slots), K(inputs), V(inputs)
            q = self.to_q(slots)  # (batch_size, num_slots, dim_slot)
            k = self.to_k(inputs)  # (batch_size, num_inputs, dim_slot)
            v = self.to_v(inputs)  # (batch_size, num_inputs, dim_slot)
            
            # 어텐션 점수 계산
            # q: (batch_size, num_slots, dim_slot)
            # k: (batch_size, num_inputs, dim_slot)
            attn = torch.einsum('bsd,bid->bsi', q, k) / math.sqrt(self.dim_slot)
            
            # 소프트맥스로 정규화 (각 입력에 대해 모든 슬롯의 가중치 합이 1)
            attn = F.softmax(attn, dim=1)  # (batch_size, num_slots, num_inputs)
            attn_weights_all.append(attn.transpose(1, 2))  # (batch_size, num_inputs, num_slots)
            
            # 가중 평균으로 슬롯 업데이트
            # attn: (batch_size, num_slots, num_inputs)
            # v: (batch_size, num_inputs, dim_slot)
            updates = torch.einsum('bsi,bid->bsd', attn, v)
            
            # GRU로 슬롯 업데이트
            slots = self.gru(
                updates.view(-1, self.dim_slot),
                slots_prev.view(-1, self.dim_slot)
            ).view(batch_size, self.num_slots, self.dim_slot)
            
            # MLP로 슬롯 개선
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        attn_weights = torch.stack(attn_weights_all, dim=1)  # (batch_size, num_iterations, num_inputs, num_slots)
        
        return slots, attn_weights


class SlotAttentionEncoder(nn.Module):
    """
    이미지를 입력으로 받아 Slot Attention으로 처리하기 위한 특징을 추출하는 인코더
    """
    
    def __init__(self, resolution, num_channels=3, hidden_dim=64):
        """
        Args:
            resolution: 입력 이미지 해상도 (정사각형 가정)
            num_channels: 입력 채널 수 (RGB=3)
            hidden_dim: 숨겨진 레이어의 차원
        """
        super().__init__()
        self.resolution = resolution
        
        # CNN 백본: 이미지를 특징 맵으로 변환
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU()
        )
        
        # 위치 인코딩 추가
        self.pos_embedding = PositionalEncoding(hidden_dim, resolution)
        
        # 특징 차원을 슬롯 차원과 맞추기 위한 변환
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, image):
        """
        Args:
            image: (batch_size, channels, height, width)
            
        Returns:
            features: (batch_size, height*width, hidden_dim) - 위치별 특징들
        """
        batch_size = image.shape[0]
        
        # CNN으로 특징 추출
        x = self.cnn(image)  # (batch_size, hidden_dim, height, width)
        
        # 위치 인코딩 추가
        x = self.pos_embedding(x)
        
        # (batch_size, hidden_dim, height, width) -> (batch_size, height*width, hidden_dim)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 정규화 및 MLP
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    2D 위치 인코딩: 각 픽셀의 위치 정보를 특징에 추가
    """
    
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        
        # 학습 가능한 위치 임베딩
        self.pos_embed = nn.Parameter(
            torch.randn(1, hidden_dim, resolution, resolution) * 0.02
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, hidden_dim, height, width)
        """
        return x + self.pos_embed


class SlotAttentionDecoder(nn.Module):
    """
    슬롯을 이용해 이미지를 재구성하는 디코더
    각 슬롯은 하나의 객체 마스크와 RGB 값을 생성
    """
    
    def __init__(self, num_slots, slot_dim, resolution, num_channels=3, hidden_dim=64):
        """
        Args:
            num_slots: 슬롯 개수
            slot_dim: 슬롯 차원
            resolution: 출력 이미지 해상도
            num_channels: 출력 채널 수
            hidden_dim: 숨겨진 레이어 차원
        """
        super().__init__()
        self.num_slots = num_slots
        self.resolution = resolution
        self.hidden_dim = hidden_dim
        
        # 슬롯을 초기 특징으로 변환
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 위치 인코딩 (디코더용)
        self.decoder_pos = nn.Parameter(
            torch.randn(1, hidden_dim, resolution, resolution) * 0.02
        )
        
        # CNN 디코더: 특징을 RGB + alpha로 변환
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, num_channels + 1, 3, stride=1, padding=1)  # RGB + alpha
        )
        
    def forward(self, slots):
        """
        Args:
            slots: (batch_size, num_slots, slot_dim)
            
        Returns:
            recons: (batch_size, num_channels, height, width) - 재구성된 이미지
            masks: (batch_size, num_slots, height, width) - 각 슬롯의 마스크
        """
        batch_size = slots.shape[0]
        
        # 각 슬롯을 독립적으로 디코딩
        slot_features = self.slot_proj(slots)  # (batch_size, num_slots, hidden_dim)
        
        # 공간 차원으로 확장
        slot_features = slot_features.view(
            batch_size * self.num_slots, self.hidden_dim, 1, 1
        ).expand(-1, -1, self.resolution, self.resolution)
        
        # 위치 인코딩 추가
        slot_features = slot_features + self.decoder_pos.expand(
            batch_size * self.num_slots, -1, -1, -1
        )
        
        # CNN 디코딩
        decoded = self.decoder_cnn(slot_features)  # (batch_size * num_slots, 4, height, width)
        
        # 슬롯별로 분리
        decoded = decoded.view(batch_size, self.num_slots, 4, self.resolution, self.resolution)
        
        # RGB와 alpha 분리
        reconstructed_slots = decoded[:, :, :3]  # (batch_size, num_slots, 3, height, width)
        alpha_masks = F.softmax(decoded[:, :, 3:], dim=1)  # (batch_size, num_slots, 1, height, width)
        
        # 알파 블렌딩으로 최종 이미지 재구성
        recons = (reconstructed_slots * alpha_masks).sum(dim=1)  # (batch_size, 3, height, width)
        masks = alpha_masks.squeeze(2)  # (batch_size, num_slots, height, width)
        
        return recons, masks


class SlotAttentionAutoEncoder(nn.Module):
    """
    완전한 Slot Attention 오토인코더
    이미지 -> 슬롯 -> 재구성 이미지
    """
    
    def __init__(self, resolution=64, num_slots=7, num_iterations=3, 
                 num_channels=3, slot_dim=64, hidden_dim=64):
        """
        Args:
            resolution: 이미지 해상도
            num_slots: 슬롯 개수
            num_iterations: 슬롯 어텐션 반복 횟수
            num_channels: 이미지 채널 수
            slot_dim: 슬롯 차원
            hidden_dim: 숨겨진 레이어 차원
        """
        super().__init__()
        
        self.encoder = SlotAttentionEncoder(resolution, num_channels, hidden_dim)
        self.slot_attention = SlotAttention(num_slots, slot_dim, hidden_dim, num_iterations)
        self.decoder = SlotAttentionDecoder(num_slots, slot_dim, resolution, num_channels, hidden_dim)
        
    def forward(self, image):
        """
        Args:
            image: (batch_size, channels, height, width)
            
        Returns:
            recons: (batch_size, channels, height, width) - 재구성된 이미지
            masks: (batch_size, num_slots, height, width) - 객체 마스크들
            slots: (batch_size, num_slots, slot_dim) - 학습된 슬롯 표현들
            attn: (batch_size, num_iterations, num_pixels, num_slots) - 어텐션 가중치들
        """
        # 1. 이미지를 특징으로 인코딩
        features = self.encoder(image)
        
        # 2. 슬롯 어텐션으로 객체 표현 학습
        slots, attn = self.slot_attention(features)
        
        # 3. 슬롯으로부터 이미지 재구성
        recons, masks = self.decoder(slots)
        
        return recons, masks, slots, attn