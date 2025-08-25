import torch as th
import torch.nn as nn
from dataclasses import asdict

from typing import Optional, Callable, Union, List, Tuple
from src.layers.block import Block
from src.configs import BlockConfig, VisualTransformerConfig
from src.models.vision_transformer import VisualTransformer


class Agregator(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        img_size: Tuple[int, int],
        out_features: Optional[int]=None,
        hiden_features: Optional[int]=None,
        apply_film: Optional[bool]=True,
        act: Callable[..., nn.Module]=nn.GELU,
        block: Callable[..., nn.Module]=Block,
        vit: Callable[..., nn.Module]=VisualTransformer,
        block_cfg: Optional[BlockConfig]=BlockConfig,
        vit_cfg: Optional[VisualTransformerConfig]=VisualTransformerConfig,
        aa_order: Optional[List[str]]=["frame", "global"],
        aa_depth: Optional[int]=4,
        register_tokens_n: Optional[int]=4,
        return_features: Optional[bool]=False,
        add_norm: Callable[..., nn.Module]=nn.LayerNorm

    ) -> None:
        

        super().__init__()
        self.order = aa_order
        self.depth = aa_depth

        if return_features:
            self.features = []
            
        self.out_features = (
            out_features
            if out_features is not None
            else embedding_dim
        )
        hiden_features = (
            hiden_features
            if hiden_features is not None
            else embedding_dim
        )

        self.vit = vit(**asdict(vit_cfg(
            img_size=img_size,
            out_features=embedding_dim,
            hiden_features=hiden_features
        )))
        self.aa_blocks = nn.ModuleDict({
            "frame": nn.ModuleList([
                block(**asdict(block_cfg(
                    in_features=embedding_dim,
                    out_features=self.out_features,
                    hiden_features=hiden_features
                ))) 
                for _ in range(self.depth)
            ]),
            "global": nn.ModuleList([
                block(**asdict(block_cfg(
                    in_features=embedding_dim,
                    out_features=self.out_features,
                    hiden_features=hiden_features
                ))) 
                for _ in range(self.depth)
            ])
        })

        if apply_film:
            self.film_layer = nn.Sequential(
                nn.Linear(hiden_features, hiden_features * 2),
                act()
            )
        
        self.camera_tokens = nn.Parameter(th.rand((1, 2, 1, embedding_dim)))
        self.register_tokens = nn.Parameter(th.rand((1, 2, register_tokens_n, embedding_dim)))
        self.add_norm = add_norm(self.out_features)

    def _apply_aa(
        self,
        tokens: th.Tensor,
        block: nn.ModuleList,
        aa_type: str,
        idx: int,
        B, S, N, C
    ):
        
        if aa_type == "frame":
            if tokens.size() != (B * S, N, C):
                tokens = tokens.view(B * S, N, C)
            
        else:
            if tokens.size() != (B, S * N, C):
                tokens = tokens.view(B, S * N, C)
        
        x = tokens
        for idx in range(idx, self.depth):

            x = block[idx](x)
            if hasattr(self, "film_layer"):
                film_features = self.film_layer(x).view(*x.size(), 2)
                shift, scale = film_features.unbind(dim=-1)
                x = scale * x + shift
        
        return x

            
            
    def _prepeare_tokens(
        self, 
        tokens: th.Tensor, 
        BS: Tuple[int, int]
    ) -> th.Tensor:
        
        B, S = BS
        quary = tokens[:, 0, ...].expand(B, 1, *tokens.size()[-2:])
        others = tokens[:, 1, ...].expand(B, S - 1, *tokens.size()[-2:])
        combined_tokens = th.cat([quary, others], dim=1)
        combined_tokens = combined_tokens.view(B * S, *combined_tokens.size()[-2:])
    

        return combined_tokens

        
    
    def forward(self, x: th.Tensor) -> None:

        patch_tokens = self.vit(x)
        B, S, N, C = patch_tokens.size()
        patch_tokens = patch_tokens.view(B * S, N, C)
        
        camera_tokens = self._prepeare_tokens(self.camera_tokens, BS=[B, S])
        register_tokens = self._prepeare_tokens(self.register_tokens, BS=[B, S])
        tokens = th.cat([
            register_tokens, 
            camera_tokens, 
            patch_tokens
        ], dim=1)
        
        N = tokens.size()[1]
        tokens = tokens.view(B, S, N, C)
        outputs = {
            "frame": None,
            "global": None
        }
        for aa_type in self.order:
            x = tokens
            prev_x = None
            block = self.aa_blocks[aa_type]
            for idx in range(self.depth):
                x = self._apply_aa(
                    tokens=x,
                    block=block,
                    aa_type=aa_type,
                    idx=idx,
                    B=B, S=S, N=N, C=C
                )
                if idx != 0:
                    x = prev_x + x
                    x = self.add_norm(x)
                
                prev_x = x
            outputs[aa_type] = x
        
        frame_tokens = outputs["frame"].view(B, S, N, self.out_features)
        global_tokens= outputs["global"].view(B, S, N, self.out_features)
        tokens = th.cat([frame_tokens, global_tokens], dim=-1)
        
        return tokens
        
        

if __name__ == "__main__":

    IMG_SIZE = (128, 128)
    PATCH_SIZE = 16
    EMBEDDING_DIM = 128
    agregator = Agregator(
        embedding_dim=EMBEDDING_DIM,
        img_size=IMG_SIZE
    )
    test = th.rand((10, 10, 3, 128, 128))
    agregator(test)
    