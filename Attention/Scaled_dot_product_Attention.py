import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

class ScaleDotProductAttention(nn.Module):
    """
    Compute Scale dot product attention
    
    Query: given sentence that we focused on (decoder)
    Key : every sentence to check realationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """
    
    def __init__ (self):
        super(ScaleDotProductAttention, self).__init__() # 초기화
        self.softmax = nn.Softmax(dim=1) # dimension=1에 대해 soft max 적용
        
    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # 1. dot product Query with Key^T to compute similarity 
        k_t = k.transpose(2, 3) # transpose 
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product 
        
        # 2. apply masking (opt)
        if mask is not None :
            score = score.masked_fill(mask==0, -10000)
            
        # 3. pass them softmax to make [0, 1] range 
        score = self.softmax(score) 
        
        # 4. multiply with value
        v = score @ v
        
        return v, score 