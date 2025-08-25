import torch as th
import torch.nn as nn
import numpy as np
from typing import Optional, Callable


class PoseEncoding(nn.Module):

    def __init__(
        self, 
        features: Optional[int]=None, 
        sequence_len: Optional[int]=None
    ) -> None:
        
        super().__init__()
        if ((features is not None) and
            (sequence_len is not None)):
            self._make_pe_buffer(features, sequence_len)
        
    def _make_pe_buffer(self, features: int, sequence_len: int) -> None:

        positions = th.arange(0, sequence_len).unsqueeze(1)
        div_term = (1 / 1000 ** (2 * th.arange(0, features) / features)).unsqueeze(dim=0)
        pose_enc = th.zeros(1, sequence_len, features)

        pose_enc[:, 0::2, :] = th.sin(positions[0::2] * div_term)
        pose_enc[:, 1::2, :] = th.cos(positions[1::2] * div_term)
        self.register_buffer("pose_enc", pose_enc)

    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, C = x.size()
        if not hasattr(self, "pose_enc"):
            self._make_pe_buffer(C, S)

        return x + self.pose_enc.repeat(B, 1, 1)
    
 
         
class Attention(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int]=None,
        hiden_features: Optional[int]=None,
        drop_rate: Optional[float]=0.32,
        mask: Optional[th.Tensor]=None,
        heads_n: int=1,
        pose_enc: Callable[..., nn.Module]=PoseEncoding,
        act: Callable[..., nn.Module]=nn.GELU,
        layer_norm: Callable[..., nn.Module]=nn.LayerNorm
    ) -> None:
        
        super().__init__()
        out_features = (
            out_features 
            if out_features is not None 
            else in_features
        )
        hiden_features = (
            hiden_features 
            if hiden_features is not None 
            else in_features
        ) 

        self.M = mask
        self.heads_n = heads_n

        self.fc_in = nn.Linear(in_features, hiden_features)
        self.fc_out = nn.Linear(hiden_features, out_features)
        self.qkv_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiden_features, hiden_features * 3),
                act(),
                nn.Dropout(drop_rate)
            ) for _ in range(self.heads_n)
        ])
        
        self.soft_act = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiden_features, hiden_features),
                act(),
                nn.Dropout(drop_rate)
            ) for _ in range(self.heads_n)
        ])
        self.pose_enc = pose_enc()
        self.layer_norm = layer_norm(hiden_features)
    
    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, C = x.size()
        pred_proj = None
        x = self.fc_in(x)
        x = self.layer_norm(x)

        for layer_idx in range(self.heads_n):
            qkv = self.qkv_proj[layer_idx](x).view(B, S, 3, C)
            q, k, v = qkv.unbind(2)
            
            qkT = (1 / C) * (q @ k.transpose(-2, -1))
            if self.M is not None:
                qkT += self.M
            
            att = qkT @ v
            att = self.soft_act(att)
            x = self.proj[layer_idx](att)

            if pred_proj is None:
                pred_proj = x
            
            else:
                x =  x + pred_proj

        x = self.fc_out(x)
        return self.pose_enc(x)
    





    


        
        


        

       
    
    

        
    
