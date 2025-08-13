import torch
import torch.nn as nn

from Transformer.model.encoder import Encoder
from Transformer.model.decoder import Decoder 

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx # 소스 문장의 PAD 토큰 ID
        self.trg_pad_idx = trg_pad_idx # 타깃 문장의 PAD 토큰 ID
        self.trg_sos_idx = trg_sos_idx # 타깃 시작 토큰(<sos>) ID (디코딩에 사용) -> 디코더 루프 시작 기준
        self.device = device # 연산 디바이스
        self.encoder = Encoder(d_model=d_model, # 임베딩/히든 차원(모든 블록의 채널 수)
                               n_head=n_head, 
                               max_len=max_len, # positional encoding에 지원하는 최대 길이
                               ffn_hidden=ffn_hidden, # FFN 내부 확장 차원 
                               enc_voc_size=enc_voc_size, # 소스 vocabulary 크기
                               drop_prob=drop_prob,       # 드롭아웃 비율 
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,  # 타깃 vocabulary 크기
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)          # [B,1,1,Ls]  인코더/크로스어텐션용 '패딩 보존(keep)' 마스크
        trg_mask = self.make_trg_mask(trg)          # [B,1,Lt,Lt] 디코더 셀프어텐션용 '패딩∧캐주얼' 결합 마스크
        enc_src = self.encoder(src, src_mask)       # [B,Ls,d_model] 인코더 출력(메모리)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)  # 보통 [B,Lt,vocab] 로짓
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (src != pad) → [B,Ls] bool: 패딩은 False, 유효토큰은 True
        # .unsqueeze(1).unsqueeze(2) → [B,1,1,Ls]로 늘려서 헤드/쿼리 길이에 브로드캐스트 가능하게 만듦
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # [B,Lt] → [B,1,Lt,1]  (패딩은 False, 유효토큰은 True)

        trg_len = trg.shape[1]
        # Lt: 타깃 길이

        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        # [Lt,Lt]  하삼각(자기 자신 및 과거만 True)인 '캐주얼(미래 가림)' 마스크
        # ※ 최신 파이토치에선 ByteTensor 대신 dtype=torch.bool 권장

        trg_mask = trg_pad_mask & trg_sub_mask
        # [B,1,Lt,1] & [Lt,Lt] → 브로드캐스트되어 [B,1,Lt,Lt]
        # 결과적으로 '패딩 보존 ∧ 미래 가림'이 모두 반영된 디코더용 마스크
        return trg_mask
