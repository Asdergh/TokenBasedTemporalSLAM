import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple
from attention import AttentionTest


class ImageTokenization(nn.Module):

    def __init__(
        self,
        dim: int,
        img_size: Tuple[int, int],
        shuffle_tokens: bool=True,
        patch_size: int=6,
        patch_mask: th.Tensor=None,
        maskin_mode: str="simple",
        binary_trashhold: float=0.32,
        maskin_epsilon: float=0.34,
        drop_rate: float=0.32,
        **conv_args
    ) -> None:
        
        super().__init__()
        W, H = img_size
        self.shuffle = shuffle_tokens
        self.patch_size = patch_size
        self.maskin_mode = maskin_mode
        self.mepsilon = maskin_epsilon

        if "out_channels" not in conv_args:
            conv_args["out_channels"] = conv_args["in_channels"]

        if "padding" not in conv_args:
            conv_args["padding"] = 1
        
        self.conv_features = nn.Sequential(
            nn.Conv2d(**conv_args),
            nn.BatchNorm2d(conv_args["out_channels"]),
            nn.Dropout2d(drop_rate),
            nn.GELU()
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(((patch_size) ** 2) * conv_args["out_channels"], dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        self._init_weigths(self.conv_features)
        self._init_weigths(self.projection)
            

        self.patches_n = (W * H) / (patch_size) ** 2
        print(self.patches_n)
        if self.patches_n % 1 != 0:
            raise ValueError(f"""
            image size must be deletable to patch_size
            curent patch_size: {patch_size}, curent img size
            {W=} x H({H=})
            """)
        self.patches_n = int(self.patches_n)

        self.M = patch_mask
        if patch_mask is None:
            self.M = th.ones(patch_size, patch_size)
        
        if maskin_mode == "binary":
            self.M[self.M > binary_trashhold] = 1.0
            self.M[self.M < binary_trashhold] = 0.0
            self.M = self.M.to(th.bool)
        
        
    def _init_weigths(self, module: nn.Module) -> None:
        for layer in module:
            if hasattr(layer, "weight"):
                nn.init.normal_(layer.weight)
    
    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, C, W, H = x.size()
        patch_tokens = []
        patched_img = []
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
                
                proj = self.projection(patch)
                proj = proj.unsqueeze(dim=2)
                patch = patch.unsqueeze(dim=2)
                patch_tokens.append(proj)
                patched_img.append(patch)
        
        tokens = th.cat(patch_tokens, dim=2)
        patched_img = th.cat(patched_img, dim=2)
        if self.shuffle:
            idx = th.argsort(th.rand(self.patches_n))
            tokens = tokens[..., idx, :]

        return (tokens, patched_img)
        

if __name__ == "__main__":

    import cv2
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")


    IMG_SIZE = (128, 128)
    PATCH_SIZE = 16
    TOKENS_DIM = 500
    ATT_DIM = 500
    HEADS = 10
    IMG_PATH = "/home/ram/Downloads/Telegram Desktop/img2/0.png"
    
    # img = cv2.imread(IMG_PATH)
    # img = cv2.resize(img, IMG_SIZE)
    # img = th.Tensor(img)
    # img = (img.to(th.float32) / 255.0).permute(2, 0, 1).unsqueeze(dim=0)
    # img = img.unsqueeze(dim=1).repeat(1, 2, 1, 1, 1)
    # print(img.size())
    img = th.rand((64, 10, 3, IMG_SIZE[0], IMG_SIZE[1]))
    img_tokenizer = ImageTokenization(
        dim=TOKENS_DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        kernel_size=(3, 3)
    )
    attention = AttentionTest(
        heads_n=HEADS,
        dim=TOKENS_DIM,
        proj_dim=ATT_DIM
    )
    
    tokens, img_patched = img_tokenizer(img)
    print(tokens.size(), img_patched.size())
    att = attention(tokens)
    print(img_patched.size(), tokens.size(), att.size())
    maps_grid = make_grid(img_patched[0, 0, ...], nrow=8).permute(1, 2, 0)
    print(maps_grid.size())

    _, axis = plt.subplots(ncols=3)
    axis[0].imshow(maps_grid.detach().numpy())
    axis[1].imshow(tokens[0, 0, ...].detach().numpy(), cmap="inferno")
    axis[2].imshow(att[0, 0, ...].detach().numpy(), cmap="coolwarm")

    plt.show()
    

        


            


