import torch as th
import torch.nn as nn

from src.layers.block import Block
from src.configs import BlockConfig
from typing import Union, Optional, Callable

class CameraHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        hiden_features: Optional[int]=32,
        head_depth: Optional[int]=3,
        apply_film: Optional[bool]=True,
        block_cfg: Optional[BlockConfig]=BlockConfig,
        norm: Callable[..., nn.Module]=nn.LayerNorm,
        act: Callable[..., nn.Module]=nn.GELU,
        out_act: Callable[..., nn.Module]=nn.GELU
    ) -> None:
        
        super().__init__()
        self.depth = head_depth

        self.blocks = nn.ModuleList([
            Block(**block_cfg(hiden_features)._asdict())
            for _ in range(head_depth)
        ])
        self.in_head = nn.Sequential(
            nn.Linear(in_features, hiden_features),
            act(),
            norm(hiden_features)
        )
        self.out_head = nn.Sequential(
            nn.Linear(hiden_features, 7),
            out_act(),
            norm(7)
        )
        self.norm = norm(hiden_features)
        if apply_film:
            self.film_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hiden_features, hiden_features * 2),
                    act()
                )
                for _ in range(head_depth)
            ])
        
    
    def forward(
        self, 
        tokens: th.Tensor,
        return_features: Optional[bool]=False
    ) -> th.Tensor:

        tokens = tokens[..., 4: 4 + 1, :].squeeze()
        B, S, C = tokens.size()
        if return_features:
            features = []
        
        x = tokens
        x = self.in_head(x)
        for idx in range(self.depth):
            x = self.blocks[idx](x)
            C = x.size()[-1]
            if hasattr(self, "film_blocks"):
                film_x = self.film_blocks[idx](x).view(B, S, C, 2)
                scale, shift = film_x.unbind(dim=-1)
                x = scale * x + shift
                x = self.norm(x)
            
            if return_features:
                features.append(x)
        
        x = self.out_head(x)
        if return_features:
            return (x, features)
        
        return x
            
        



if __name__ == "__main__":

    from src.models.agregator import Agregator
    
    IMG_SIZE = (128, 128)
    PATCH_SIZE = 16
    TOKENS_DIM = 32
    EMBEDDING_DIM = 64
    agregator = Agregator(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hiden_features=EMBEDDING_DIM,
        out_features=TOKENS_DIM
    )
    cam_head = CameraHead(TOKENS_DIM * 2)
    test = th.normal(0, 1, (10, 2, 3, 128, 128))
    tokens = agregator(test)
    cam_out = cam_head(tokens)

    print(cam_out.size())
        