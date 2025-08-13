import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

# in study(basic) ===================================================================
class ScaleDotProductAttention(nn.Module):
    """
    Compute Scale dot product attention
    
    Query: given sentence that we focused on (decoder)
    Key : every sentence to check realationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """
    # softmax, relu와 같이 파라미터가 없는 것들은 init에서 정의하지 않고 forward에서 F.softmax로 사용 가능
    # Linear, Conv2d와 같이 학습할 파라미터가 있는건 init에서 정의해야함
    
    def __init__ (self, dropout=0.1, temperature=1.0): 
        # 클래스 생성 시 한 번만 호출, 모델 구조 정의 (레이어(파라미터), 하이퍼파라미터, 상수값 등 저장), 연산X
        super(ScaleDotProductAttention, self).__init__() # 초기화
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature # 학습 가능한 temperature 
        
        # self.softmax = nn.Softmax(dim=-1) # 마지막 dimension에 대해 softmax 적용
        
        
    def forward(self, Q, K, V, mask=None, e=1e-12): 
        # 모델 호출 시마다 실행, 실제 데이터 연산 수행
        
        # input is 4 dimension tensor [batch size, head, length, d_tensor]
        # batch size = 배치사이즈, head=multi-head attention 헤드 수, length = 토큰 개수(문장 길이), d_tensor = 한 헤드가 보는 임베딩 차원 수
        # 이미지일 경우 length = patch 개수, d_model = patch 임베딩 후 feature 벡터 크기 등
        # batch_size, head, length, d_tensor = K.size() 
        # => 입력 텐서가 무조건 [B, H, L, D] 모양일 경우
        
        d_k = Q.size(-1) # 입력 텐서가 [...,D] 꼴이면 앞의 차원이 무엇이든 신경 안 씀, shape에 유연
        # 일반적인 Scaled Dot-Product Attention에서는 Q와 K의 마지막 차원 크기(d_k)가 동일하기 때문에 가능
        
        # 1. dot product Query with Key^T to compute similarity 
        # k_t = K.transpose(2, 3) # transpose, 차원 위치 하드코딩 ([B, H, L, D] -> [B, H, D, L])
        # score = (Q @ k_t) / math.sqrt(d_tensor) # scaled dot product  
        # (Q @ k_t) = [B, H, L, D] @ [B, H, D, L] = [B, H, L, L] (@, matmul은 마지막 두 차원을 행렬 곱셈 대상으로 보고, 나머지는 브로드 캐스팅)
        
        score = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * self.temperature)
        # transpose(-2, -1) # 마지막 두 축 기준 transpose, matmul == @ 
        # self.temperature : temperature 조절 기능 추가 (Attention의 "날카로움" 조절(opt))
        
        # 2. apply masking (opt)
        if mask is not None :
            score = score.masked_fill(mask==0, -1e9) 
            # -100000는 Softmax에서 완전히 0에 가까운 확률을 만들려고 사용. 더 큰 음수값(-1e9) 사용 가능
            
        # 3. pass them softmax to make [0, 1] range 
        # score = self.softmax(score) # init에서 softmax 정의시 사용
        score = F.softmax(score, dim=-1) # 마지막 차원 기준 확률 분포 생성
        # QK^t -> [B, H, L_q, L_k] : 각 쿼리 토큰이 키 토큰과 얼마나 유사한지 담음 
        # 따라서, 마지막 차원(L_k)에 대해 softmax해야 각 쿼리 토큰별로 키 전체에 대한 확률 분포 나옴 
        
        score = self.dropout(score) # dropout 적용
        
        # 4. multiply with value
        # V = score @ V
        output = torch.matmul(score, V) # @ == matmul(), 유지보수를 위해 output에 저장. 
        
        return output, score 
