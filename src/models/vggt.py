import torch as th
import torch.nn as nn

from typing import (
    Tuple,
    Optional
)
from src.configs import (
    AgregatorConfig,
    DenseHeadConfig,
    CamHeadConfig
)
from src.models.agregator import Agregator
from src.heads.camera_head import CameraHead
from src.heads.dense_head import DptHead



class VggtModel(nn.Module):

    def __init__(
        self,
        img_size: Optional[Tuple[int, int]]=(128, 128),
        patch_size: Optional[int]=16,
        token_emd_dim: Optional[int]=32,
        token_dim: Optional[int]=64,
        agregator_cfg: Optional[AgregatorConfig]=AgregatorConfig,
        dpt_pcd_cfg: Optional[DenseHeadConfig]=DenseHeadConfig,
        dpt_depth_cfg: Optional[DenseHeadConfig]=DenseHeadConfig,
        cam_head_cfg: Optional[CamHeadConfig]=CamHeadConfig
    ) -> None:
        
        super().__init__()
        agregator_cfg = agregator_cfg(
            img_size=img_size,
            embedding_dim=token_emd_dim,
            out_features=token_dim
        )._asdict()
        self.agregator = Agregator(**agregator_cfg)

        if dpt_depth_cfg is not None:
            dpt_depth_cfg = dpt_depth_cfg(
                patch_size=patch_size,
                img_size=img_size,
                in_features=(token_dim * 2),
                out_features=2
            )._asdict()
            self.dpt_depth_head = DptHead(**dpt_depth_cfg)
        
        if dpt_pcd_cfg is not None:
            dpt_pcd_cfg = dpt_pcd_cfg(
                patch_size=patch_size,
                img_size=img_size,
                in_features=(token_dim * 2),
                out_features=4
            )._asdict()
            self.dpt_pcd_head = DptHead(**dpt_pcd_cfg)

        if cam_head_cfg is not None:
            cam_head_cfg = cam_head_cfg(in_features=(token_dim * 2))._asdict()
            self.cam_head = CameraHead(**cam_head_cfg)
        
        
    
    def forward(self, imgs: th.Tensor) -> dict:

        B, S, C, W, H = imgs.size()
        prediction = {}
        tokens = self.agregator(imgs)
        prediction["tokens"] = {
            "reg_tokens": tokens[..., 0:4, :],
            "cam_tokens": tokens[..., 4:4 + 1, :],
            "patch_tokens": tokens[..., 4 + 1:, :]
        }

        if hasattr(self, "dpt_depth_head"):
            depth_pred = self.dpt_depth_head(tokens)
            depth_map, conf = depth_pred.unbind(dim=2)
            prediction["depth"] = depth_map
            prediction["depth_conf"] = conf
        
        if hasattr(self, "dpt_pcd_head"):
            pcd_pred = self.dpt_depth_head(tokens)
            pcd_map, conf = depth_pred.unbind(dim=2)
            prediction["pcd"] = pcd_pred
            prediction["pcd_conf"] = pcd_map
        
        if hasattr(self, "cam_head"):
            cam_poses = self.cam_head(tokens)
            prediction["cam_poses"] = cam_poses

        return prediction


if __name__ == "__main__":

    test = th.normal(0, 1, (10, 2, 3, 128, 128))
    vggt = VggtModel()
    print(vggt(test).keys())
    

        