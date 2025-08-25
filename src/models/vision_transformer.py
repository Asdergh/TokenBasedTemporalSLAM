import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple
from src.layers.attention import Attention
from typing import Optional, Union, Callable


class VisualTransformer(nn.Module):

    def __init__(
        self,
        out_features: int,
        hiden_features: Optional[int],
        img_size: Tuple[int, int],
        shuffle_tokens: Optional[bool]=True,
        patch_size: Optional[int]=16,
        patch_mask: Optional[th.Tensor]=None,
        maskin_mode: Optional[str]="simple",
        binary_trashhold: Optional[float]=0.32,
        maskin_epsilon: Optional[float]=0.34,
        drop_rate: Optional[float]=0.32,
        act_fc_part: Callable[..., nn.Module]=nn.GELU,
        act_conv_part: Callable[..., nn.Module]=nn.GELU,
        layer_norm_fc: Callable[..., nn.Module]=nn.LayerNorm,
        layer_norm_conv: Callable[..., nn.Module]=nn.BatchNorm2d
    ) -> None:
        
        super().__init__()
        W, H = img_size
        hiden_features = (
            hiden_features
            if hiden_features is not None
            else out_features
        )

        self.shuffle = shuffle_tokens
        self.patch_size = patch_size
        self.maskin_mode = maskin_mode
        self.mepsilon = maskin_epsilon
        
        self.patches_n = (W * H) / (patch_size) ** 2
        if self.patches_n % 1 != 0:
            raise ValueError(f"""
            image size must be deletable to patch_size
            curent patch_size: {patch_size}, curent img size
            {W=} x H({H=})
            """)
        self.patches_n = int(self.patches_n)

    
        self.conv_features = nn.Sequential(
            nn.Conv2d(3, 3, (3, 3), 1, 1),
            layer_norm_conv(3),
            nn.Dropout2d(drop_rate),
            act_conv_part()
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(((patch_size) ** 2) * 3, hiden_features),
            layer_norm_fc(hiden_features),
            act_fc_part(),
            nn.Dropout(drop_rate)
        )
        self.fc2 = nn.Linear(hiden_features, out_features)
        self.M = patch_mask
        if patch_mask is None:
            self.M = th.ones(patch_size, patch_size)
        
        if maskin_mode == "binary":
            self.M[self.M > binary_trashhold] = 1.0
            self.M[self.M < binary_trashhold] = 0.0
            self.M = self.M.to(th.bool)
        
        self._init_weigths(self.conv_features)
        self._init_weigths(self.fc1)
        
        
    def _init_weigths(self, module: nn.Module) -> None:
        for layer in module:
            if hasattr(layer, "weight"):
                nn.init.normal_(layer.weight)
    
    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, C, W, H = x.size()
        patch_tokens = []
        conv_features = []
        for frame_idx in range(S):
            frame = x[:, frame_idx, ...]
            conv_x = self.conv_features(frame)
            conv_x = conv_x.unsqueeze(dim=1)
            conv_features.append(conv_x)
        conv_features = th.cat(conv_features, dim=1)

        patch_per_row = int(np.sqrt(self.patches_n))
        for l_idx in range(patch_per_row):
            for r_idx in range(patch_per_row):

                
                winLL = l_idx * self.patch_size
                winLR = (l_idx + 1) * self.patch_size
                winRL = r_idx * self.patch_size
                winRR = (r_idx + 1) * self.patch_size
                patch = conv_features[..., winLL: winLR, winRL: winRR]

                if th.rand(1) > self.mepsilon:
                    patch *= self.M
            
                x = self.fc1(patch)
                x = x.unsqueeze(dim=2)
                patch_tokens.append(x)
        
        tokens = th.cat(patch_tokens, dim=2)
        if self.shuffle:
            idx = th.argsort(th.rand(self.patches_n))
            tokens = tokens[..., idx, :]

        tokens = self.fc2(tokens)
        return tokens
        


        


            


