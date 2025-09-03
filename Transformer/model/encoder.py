import torch 
import torch.nn as nn

from Transformer.block.encoder_layer import EncoderLayer
from Transformer.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module): # Embedding -> N개의 Encoder layer stack -> 출력
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device) # 입력 x: [B, L] (Long, 토큰 ID) / 출력: [B, L, d_model]

        # nn.ModuleList : 여러 개의 서브 모듈을 리스트처럼 보관하며 등록하는 컨테이너 
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)]) # n_layers번 stack, 각 레이어 입력/출력 shape 동일: [B, L, d_model]
        

    def forward(self, x, src_mask):
        x = self.emb(x)  # [B, L, d_model]

        for layer in self.layers:  
            x = layer(x, src_mask) # [B, L, d_model]

        return x # "memory"로 디코더 cross-attn에 들어감