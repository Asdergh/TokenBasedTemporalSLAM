import torch as th
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Optional, Callable, Tuple
from src.layers.attention import PoseEncoding, Attention
from src.layers.mlp import Mlp

@dataclass
class AttentionConfig:
    in_features: int
    out_features: Optional[int]=None
    hiden_features: Optional[int]=None
    drop_rate: Optional[float]=0.32
    mask: Optional[th.Tensor]=None
    heads_n: int=1
    pose_enc: Callable[..., nn.Module]=PoseEncoding
    act: Callable[..., nn.Module]=nn.GELU
    layer_norm: Callable[..., nn.Module]=nn.LayerNorm

@dataclass 
class MlpConfig:
    in_features: int
    out_features: Optional[int]=None
    hiden_features: Optional[int]=None
    act: Callable[..., nn.Module]=nn.GELU
    drop_rate: Optional[float]=0.32
    layer_norm_fc1: Callable[..., nn.Module]=nn.LayerNorm
    layer_norm_fc2: Callable[..., nn.Module]=nn.LayerNorm

@dataclass 
class VisualTransformerConfig:
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

@dataclass
class BlockConfig:
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


    