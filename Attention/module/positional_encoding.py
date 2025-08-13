import torch 
import torch.nn as nn
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        Args:
            d_model (_type_): dimension of model
            max_len (_type_): max sequence length
            device (_type_): hardware device setting
        """
        
        super(PositionalEncoding, self).__init__()
        
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # we don't need to compute gradient
        
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # i means index of d_model (e.g., embedding size = 50, 'i' = [0, 50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 **(_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words 
        
    def forward(self, x) :
        # self.encoding
        
        batch_size, seq_len = x.size()
        
        return self.encoding[:seq_len, :]