import torch as th
import torch.nn as nn


from typing import Optional, Union, Callable, Tuple

from src.layers.block import Block
from src.layers.mlp import Mlp
from src.configs import BlockConfig, MlpConfig




class DptHead(nn.Module):

    def __init__(
        self,
        patch_size: int,
        img_size: Tuple[int, int],
        in_features: int,
        hiden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        ressable_act: Callable[..., nn.Module]=nn.GELU,
        fusion_act: Callable[..., nn.Module]=nn.GELU,
        fusion_norm: Callable[..., nn.Module]=nn.BatchNorm2d,
        ressable_norm: Callable[..., nn.Module]=nn.BatchNorm2d,
        block_cfg: Optional[BlockConfig]=BlockConfig,
        head_depth: Optional[int]=4,
        return_features: Optional[bool]=False
    ) -> None:
        

        super().__init__()
        hiden_features = (
            hiden_features
            if hiden_features is not None
            else in_features
        )

        self.w, self.h = img_size
        self.C_out = out_features
        self.depth = head_depth
        self.patches_n = int((img_size[0] * img_size[1]) / (patch_size ** 2))
        self.ppr = int(th.sqrt(th.tensor(self.patches_n)).item())
        
        if return_features:
            self.features = []

        self.in_head = self.out_head = nn.Sequential(
            nn.Conv2d(in_features, hiden_features, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hiden_features)
        )
        self.fusion_units = self._get_conv(
            n=self.depth,
            features=hiden_features,
            act=fusion_act,
            norm=fusion_norm
        ) 
        self.ressable_units = self._get_conv(
            n=self.depth,
            features=hiden_features,
            act=ressable_act,
            norm=ressable_norm
        ) 
        self.ressable_res = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=hiden_features, 
                out_channels=hiden_features, 
                kernel_size=(2 ** (idx + 1)), 
                stride=2 ** (idx + 1)
            )
            for idx in range(self.depth)
        ])
        self.transformer_blocks = nn.ModuleList(
            Block(**block_cfg(in_features)._asdict())
            for _ in range(self.depth)
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(hiden_features, out_features, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_features)
        )
    
    def _get_conv(
        self, 
        n: int, features: int, 
        act: Callable, 
        norm: Callable,
        drop_rate: Optional[float]=0.32
    ) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, (3, 3), 1, 1),
                act(),
                norm(features),
                nn.Dropout2d(drop_rate)
            )
            for _ in range(n)
        ])
    

    def forward(self, tokens: th.Tensor) -> th.Tensor:

        tokens = tokens[..., -self.patches_n:, :]
        B, S, N, C = tokens.size()
        tokens = tokens.view(B * S, N, C)

        tf_features = []
        x = tokens

        for idx in range(self.depth):

            x = self.transformer_blocks[idx](x)
            if idx == 0:
                C = x.size()[-1]

            patch_x = x.view(B * S, C, self.ppr, self.ppr)
            
            patch_x = self.in_head(patch_x)
            patch_x = self.ressable_units[idx](patch_x)
            patch_x = self.ressable_res[idx](patch_x)
            tf_features.append(patch_x)
        
        prev_x = None
        for idx in range(self.depth):
            
            skip_x = tf_features[idx]
            x = self.fusion_units[idx](skip_x)
            if prev_x is not None:
                x = x + prev_x

            prev_x = self.ressable_res[0](x)
            if hasattr(self, "features"):
                self.features.append(x)
  
        if hasattr(self, "features"):
            return (self.out_head(x), self.features)
        
        return self.out_head(x).view(B, S, self.C_out, self.w, self.h)

            
        

if __name__ == "__main__":

    from src.models.agregator import Agregator

    IMG_SIZE = (128, 128)
    EMBEDDING_DIM = 32
    TOKENS_DIM = 32
    DPT_HIDEN_FEATURES = 32
    DPT_OUT_FEATURES = 1
    
    agregator = Agregator(
        img_size=IMG_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hiden_features=32,
        out_features=TOKENS_DIM
    )
    dpt_head = DptHead(
        img_size=IMG_SIZE,
        patch_size=16,
        in_features=TOKENS_DIM * 2,
        hiden_features=32,
        out_features=3,
        head_depth=4
    )
    

  






