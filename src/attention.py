import torch as th
import torch.nn as nn
import numpy as np



class PoseEncoding(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, N, d = x.size()
        _buffer = th.zeros(B, S, N, d)

        if d % 2 == 0:
            sin_d = cos_d = d // 2
        
        else:
            sin_d = ((d - 1) // 2) + 1
            cos_d = (d - 1) // 2

        sin_ape = th.Tensor([
            [
                (np.sin(idx / (1000 * (2 * (k_idx / d)) + 1e-2)))
                for k_idx in range(sin_d)
            ]for idx in range(N)
        ])
        cos_ape =  th.Tensor([
            [
                (np.cos(idx / (1000 * (2 * (k_idx / d)) + 1e-2)))
                for k_idx in range(cos_d)
            ]for idx in range(N)
        ])
        _buffer[..., 0::2] = sin_ape
        _buffer[..., 1::2] = cos_ape

        return x + _buffer
        
         
class AttentionTest(nn.Module):

    def __init__(
        self,
        dim: int,
        proj_dim: int,
        qkv_drop: float=0.32,
        proj_drop: float=0.32,
        mask: th.Tensor=None,
        heads_n: int=1,
        epsilon: float=1e-2,
        apply_ape: bool=True
    ) -> None:
        
        super().__init__()
        self.M = mask
        self.heads_n = heads_n
        self.epsilon = epsilon
        self.apply_ape = apply_ape
        if self.apply_ape:
            self.ape = PoseEncoding()

        self.qkv_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 3),
                nn.SELU(),
                nn.Dropout(qkv_drop)
            ) for _ in range(self.heads_n)
        ])
        
        self.soft_act = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, proj_dim),
                nn.GELU(),
                nn.Dropout(proj_drop)
            ) for _ in range(self.heads_n)
        ])
    
    def __call__(self, x: th.Tensor) -> th.Tensor:

        B, S, N, d = x.size()
        pred_proj = None
        for layer_idx in range(self.heads_n):
            qkv = self.qkv_proj[layer_idx](x).view(B, S, N, 3, d)
            q, k, v = qkv.unbind(3)
            
            qkT = (1 / d) * (q @ k.transpose(-2, -1))
            if self.M is not None:
                qkT += self.M
            
            att = qkT @ v
            att = self.soft_act(att)
            proj = self.proj[layer_idx](att)

            if pred_proj is None:
                pred_proj = proj
            
            else:
                proj =  proj + self.epsilon * pred_proj

        if self.apply_ape:
            proj = self.ape(proj)

        return proj




if __name__ == "__main__":


    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    B = 10
    N = 345
    F = 345
    D = 345
    test = th.rand(B, N, F)
    ape = PoseEncoding()
    attention = AttentionTest(
        heads_n=10,
        dim=F,
        proj_dim=D,
        proj_drop=0.25,
        qkv_activation="relu",
        qkv_drop=0.25
    )
    test_out = attention(test)
    tokens = ape(test_out)
    _, axis = plt.subplots(ncols=2)
    axis[0].imshow(test_out[0, ...].detach().numpy(), cmap="twilight")
    axis[1].imshow(tokens[0, ...].detach().numpy(), cmap="twilight")
    plt.show()
    # print(test_out.size(), tokens.size())
    


        
        


        

       
    
    

        
    
