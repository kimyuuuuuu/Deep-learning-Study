import torch 
import torch.nn as nn
import torch 

from module.Scaled_dot_product_Attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads 
        self.d_model = d_model                # d_model = H * D (head수 * head당 차원(임베딩/히든 벡터 차원 수))
        self.head_dim = d_model // num_heads  

        # ScaleDotProductAttention이 softmax 후 dropout을 지원하도록 수정하길 권장
        self.attention = ScaleDotProductAttention()  
        self.w_q = nn.Linear(d_model, d_model) # nn.Linear(in_feature, out_feature) : 가중치/바이어스 파라미터를 내부에 들고 있는 레이어 하나 제작
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, return_attention=False):
        # 1) proj
        q = self.w_q(Q)
        k = self.w_k(K)
        v = self.w_v(V)

        # 2) split heads: [B,L,d_model] -> [B,H,L,D]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3) attention ([B,H,L,D], [B,H,L,L])
        out, attn = self.attention(q, k, v, mask=mask)

        # (선택) out dropout: 보통은 attn-weights dropout이 정석이지만,
        # attention 모듈에서 안 한다면 여기 출력에 가볍게 넣을 수도 있음
        # out = self.dropout(out)

        # 4) concat + final proj
        out = self.concat(out)                 # [B,L,d_model]
        out = self.w_concat(out)               # [B,L,d_model]
        
        return out, attn

    def split(self, tensor):
        # batch_size, length, d_model = tensor.size()
        
        # tensor: [B, L, d_model] -> [B, H, L, D]
        B, L, _ = tensor.size() # size 
        
        tensor = tensor.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) 
        # reshape(B, L, ..) = [B, L, H, D]
        # .transpose(1, 2) = [B, H, L, D] => Head차원을 앞쪽에 두면 연산이 깔끔해지고 헤드별 점곱이 자연스럽게 진행
        # (Q @ Kᵀ) = [B, H, L, D] @ [B, H, D, L] → [B, H, L, L]
        return tensor

    def concat(self, tensor):
        # [B, H, L, D] -> [B, L, d_model]
        B, H, L, D = tensor.size()
        return tensor.transpose(1, 2).contiguous().reshape(B, L, H * D)
