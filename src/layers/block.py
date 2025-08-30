import torch as th
import torch.nn as nn

from src.layers.attention import Attention, PoseEncoding
from src.layers.mlp import Mlp
from typing import Union, Tuple, Callable, Optional
from src.configs import AttentionConfig, MlpConfig


class Block(nn.Module):

    def __init__(
        self,
        in_features: int,
        hiden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        block_depth: Optional[int]=1,
        pad_rate: Optional[float]=0.32,
        att_fn: Callable[..., nn.Module]=Attention,
        mlp_fn: Callable[..., nn.Module]=Mlp,
        att_act: Callable[..., nn.Module]=nn.GELU,
        fc_act: Callable[..., nn.Module]=nn.GELU,
        add_norm: Callable[..., nn.Module]=nn.LayerNorm,
        add_bias: Optional[bool]=True,
        att_cfg: Optional[AttentionConfig]=AttentionConfig,
        mlp_cfg: Optional[MlpConfig]=MlpConfig,
        return_features: Optional[bool]=False
    ) -> None:
        
        super().__init__()
        if return_features:
            self.features = []

        hiden_features = (
            hiden_features
            if hiden_features is not None
            else in_features
        )
        out_features = (
            out_features
            if out_features is not None
            else in_features
        )
        self.depth = block_depth
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                mlp_fn(**mlp_cfg(hiden_features)._asdict()),
                nn.Dropout(pad_rate),
                fc_act()
            )
            for _ in range(self.depth)
        ])
        self.att_layers = nn.ModuleList([
            nn.Sequential(
                att_fn(**att_cfg(hiden_features)._asdict()),
                nn.Dropout(pad_rate),
                att_act()
            )
            for _ in range(self.depth)
        ])
        self.add_norm = add_norm(hiden_features)
        self.fc_in = nn.Linear(in_features, hiden_features)
        self.fc_out = nn.Linear(hiden_features, out_features)

        self.bias = 1e-2
        if add_bias:
            self.bias = nn.Parameter(th.tensor(self.bias))

        self._init_weigths(self.mlp_layers)
        self._init_weigths(self.att_layers)

    def _init_weigths(self, module: nn.Module) -> None:
        for layer in module:
            if hasattr(layer, "weight"):
                nn.init.normal_(layer.weight) 
    

    def forward(self, x: th.Tensor) -> th.Tensor:

        x = self.fc_in(x)
        for idx in range(self.depth):
            x = self.mlp_layers[idx](x)
            x = self.att_layers[idx](x) + (self.bias * x)
            x = self.add_norm(x)

            if hasattr(self, "features"):
                self.features.append(x)

        x = self.fc_out(x)
        if hasattr(self, "features"):
            return (x, self.features)
        
        return x


