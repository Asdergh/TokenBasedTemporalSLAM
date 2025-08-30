import torch as th
import torch.nn as nn

from typing import (
    Union, 
    Optional, 
    Callable, 
    Tuple, 
    List,
    NamedTuple
)
from src.layers.attention import PoseEncoding, Attention
from src.layers.mlp import Mlp
from src.models.vision_transformer import VisualTransformer


class AttentionConfig(NamedTuple):
    in_features: int
    out_features: Optional[int]=None
    hiden_features: Optional[int]=None
    drop_rate: Optional[float]=0.32
    mask: Optional[th.Tensor]=None
    heads_n: int=1
    pose_enc: Callable[..., nn.Module]=PoseEncoding
    act: Callable[..., nn.Module]=nn.GELU
    layer_norm: Callable[..., nn.Module]=nn.LayerNorm

 
class MlpConfig(NamedTuple):
    in_features: int
    out_features: Optional[int]=None
    hiden_features: Optional[int]=None
    act: Callable[..., nn.Module]=nn.GELU
    drop_rate: Optional[float]=0.32
    layer_norm_fc1: Callable[..., nn.Module]=nn.LayerNorm
    layer_norm_fc2: Callable[..., nn.Module]=nn.LayerNorm

 
class VisualTransformerConfig(NamedTuple):
    out_features: int
    hiden_features: Optional[int]
    img_size: Tuple[int, int]
    shuffle_tokens: Optional[bool]=True
    patch_size: Optional[int]=16
    patch_mask: Optional[th.Tensor]=None
    maskin_mode: Optional[str]="simple"
    binary_trashhold: Optional[float]=0.32
    maskin_epsilon: Optional[float]=0.34
    drop_rate: Optional[float]=0.32
    act_fc_part: Callable[..., nn.Module]=nn.GELU
    act_conv_part: Callable[..., nn.Module]=nn.GELU
    layer_norm_fc: Callable[..., nn.Module]=nn.LayerNorm
    layer_norm_conv: Callable[..., nn.Module]=nn.BatchNorm2d


class BlockConfig(NamedTuple):
    in_features: int
    hiden_features: Optional[int]=None
    out_features: Optional[int]=None
    block_depth: Optional[int]=1
    pad_rate: Optional[float]=0.32
    att_fn: Callable[..., nn.Module]=Attention
    mlp_fn: Callable[..., nn.Module]=Mlp
    att_act: Callable[..., nn.Module]=nn.GELU
    fc_act: Callable[..., nn.Module]=nn.GELU
    add_norm: Callable[..., nn.Module]=nn.LayerNorm
    add_bias: Optional[bool]=True
    att_cfg: Optional[AttentionConfig]=AttentionConfig
    mlp_cfg: Optional[MlpConfig]=MlpConfig



class AgregatorConfig(NamedTuple):
    embedding_dim: int=32
    img_size: Tuple[int, int]=(128, 128)
    patch_size: int=16
    out_features: Optional[int]=None
    hiden_features: Optional[int]=None
    apply_film: Optional[bool]=True
    act: Callable[..., nn.Module]=nn.GELU
    vit: Callable[..., nn.Module]=VisualTransformer
    in_head: Callable[..., nn.Module]=Mlp
    out_head: Callable[..., nn.Module]=Mlp
    block_cfg: Optional[BlockConfig]=BlockConfig
    vit_cfg: Optional[VisualTransformerConfig]=VisualTransformerConfig
    in_head_cfg: Optional[MlpConfig]=MlpConfig
    out_head_cfg: Optional[MlpConfig]=MlpConfig
    aa_order: Optional[List[str]]=["frame", "global"]
    aa_depth: Optional[int]=4
    register_tokens_n: Optional[int]=4
    return_features: Optional[bool]=False
    add_norm: Callable[..., nn.Module]=nn.LayerNorm


 
class DenseHeadConfig(NamedTuple):
    patch_size: int
    img_size: Tuple[int, int]
    in_features: int
    hiden_features: Optional[int]=None
    out_features: Optional[int]=None
    ressable_act: Callable[..., nn.Module]=nn.GELU
    fusion_act: Callable[..., nn.Module]=nn.GELU
    fusion_norm: Callable[..., nn.Module]=nn.BatchNorm2d
    ressable_norm: Callable[..., nn.Module]=nn.BatchNorm2d
    block_cfg: Optional[BlockConfig]=BlockConfig
    head_depth: Optional[int]=4
    return_features: Optional[bool]=False



class CamHeadConfig(NamedTuple):
    in_features: int
    hiden_features: Optional[int]=32
    head_depth: Optional[int]=3
    apply_film: Optional[bool]=True
    block_cfg: Optional[BlockConfig]=BlockConfig
    norm: Callable[..., nn.Module]=nn.LayerNorm
    act: Callable[..., nn.Module]=nn.GELU
    out_act: Callable[..., nn.Module]=nn.GELU