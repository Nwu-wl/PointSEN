"""
PointSEN
"""
from re import I
from typing import List, Type
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import copy
import numpy as np

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type
        self.adagn = nn.BatchNorm1d(channels1[-1], affine=False)

    def forward(self, pf, pe) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        ident = f
        # grouping
        dp, fj = self.grouper(p, p, f)
        # pe + fj
        f = pe + fj
        f = self.adagn(self.pool(f)-ident)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f
import math

def cart2sph(xyz, normalized=False):
    """
    xyz: (B*, G, 3) or (..., 3)
    returns rho, theta in [0,pi], phi in [-pi,pi]
    if normalized=True, returns theta in [0,1], phi in [0,1]
    """
    rho = torch.linalg.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-12)
    theta = torch.acos((xyz[..., 2:3] / rho).clamp(-1 + 1e-6, 1 - 1e-6))
    phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1])  # [-pi, pi]
    if normalized:
        theta = theta / math.pi
        phi = phi / (2 * math.pi) + 0.5
    return torch.cat([rho, theta, phi], dim=-1)

def _batched_rodrigues_align_to_z(a, eps=1e-8):
    """
    Rotate vectors so that a -> z_hat = [0,0,1].
    a: (B*n, 3) unit vectors
    returns R: (B*n, 3, 3)
    """
    Bn = a.shape[0]
    device = a.device
    z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(Bn, 3)

    a = a / (a.norm(dim=-1, keepdim=True) + eps)

    v = torch.cross(a, z, dim=-1)                    # rotation axis (not unit if a//z)
    s = torch.linalg.norm(v, dim=-1, keepdim=True)   # |v| = sin(theta)
    c = torch.sum(a * z, dim=-1, keepdim=True)       # cos(theta)

    # handle a ~ ±z
    near_zero = (s < eps).squeeze(-1) & ((c > 0).squeeze(-1))   # already aligned
    near_pi   = (s < eps).squeeze(-1) & ((c < 0).squeeze(-1))   # opposite

    # for near_pi, pick any orth vector as axis
    if near_pi.any():
        ref = torch.where(
            (a.abs()[:, 0:1] < 0.9),
            torch.tensor([1.0, 0.0, 0.0], device=device).expand(Bn, 3),
            torch.tensor([0.0, 1.0, 0.0], device=device).expand(Bn, 3),
        )
        v_alt = torch.cross(a, ref, dim=-1)
        v_alt = v_alt / (v_alt.norm(dim=-1, keepdim=True) + eps)
        v = torch.where(near_pi.unsqueeze(-1), v_alt, v)

    # unit axis
    k = v / (v.norm(dim=-1, keepdim=True) + eps)
    K = torch.zeros(Bn, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2],  k[:, 1]
    K[:, 1, 0], K[:, 1, 2] =  k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1],  k[:, 0]

    sinθ = s
    cosθ = c
    I = torch.eye(3, device=device).unsqueeze(0).expand(Bn, 3, 3)
    R = I + K * sinθ.unsqueeze(-1) + (K @ K) * (1.0 - cosθ.unsqueeze(-1))

    # if already aligned (a≈+z), use identity
    if near_zero.any():
        R[near_zero] = I[near_zero]
    return R
def roll_pad(x, kernel_h=5, kernel_w=5):
    """
    x: (B, D, H, W)
    theta (H) -> zero-pad by (kh-1)//2
    phi   (W) -> circular-pad by (kw-1)//2
    """
    B, D, H, W = x.shape
    pad_h = (kernel_h - 1) // 2
    pad_w = (kernel_w - 1) // 2

    # zero-pad along theta (top/bottom)
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h))  # (left,right,top,bottom)

    # circular-pad along phi (left/right)
    if pad_w > 0:
        left  = x[..., :, -pad_w:]
        right = x[..., :,  :pad_w]
        x = torch.cat([left, x, right], dim=-1)

    return x
def compute_h(rho, rho_max, eps=1e-12):
    """
    rho:   (B*n, G, 1)
    rho_max: (B*n, 1, 1), per-anchor max
    returns h in (0.5, 1.0]
    """
    r = rho / (2.0 * (rho_max + eps))
    h = F.relu(0.5 - r) + 0.5
    return h


def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def _3d22d_constructor(dp, gf=None, h=9, w=9, feature=None):
    """
    dp: (B, 3, N, G)  relative coords from anchor to its G neighbors
    Returns:
        freq: (B*N, H, W, D)   with D = 4 here: [dx, dy, dz, h]
        uv:   (B*N*G, 2)       integer indices (theta_id, phi_id)
        hval: (B*N*G, 1)       h factor (for debug/inspection)
    """
    device = dp.device
    B, _, N, G = dp.shape

    # (B, N, G, 3)
    dp_ng3 = dp.permute(0, 2, 3, 1).contiguous()

    # --- find farthest neighbor per anchor as reference axis di ---
    rho = torch.linalg.norm(dp_ng3, dim=-1)                # (B, N, G)
    far_idx = torch.argmax(rho, dim=-1, keepdim=True)      # (B, N, 1)
    idx = far_idx.unsqueeze(-1).expand(-1, -1, -1, 3)      # (B, N, 1, 3)
    di = torch.gather(dp_ng3, 2, idx).squeeze(2)           # (B, N, 3)
    di = di / (di.norm(dim=-1, keepdim=True) + 1e-12)

    # --- rotate all dp so that di aligns with z ---
    # reshape to (B*N, G, 3)
    dp_bn_g3 = dp_ng3.view(B * N, G, 3)
    di_bn_3  = di.view(B * N, 3)

    R = _batched_rodrigues_align_to_z(di_bn_3)             # (B*N,3,3)
    # (B*N, G, 3) -> (B*N, G, 3)  v' = v @ R^T
    dp_rot = torch.matmul(dp_bn_g3, R.transpose(1, 2))

    # --- spherical coords (un-normalized angles) ---
    sph = cart2sph(dp_rot, normalized=False)               # (B*N, G, 3)
    rho_bn_g1  = sph[..., 0:1]                             # (B*N, G, 1)
    theta_bn_g1= sph[..., 1:2]                             # [0, pi]
    phi_bn_g1  = sph[..., 2:3]                             # [-pi, pi]
    # map phi to [0, 2pi)
    phi_bn_g1 = (phi_bn_g1 + 2*math.pi) % (2*math.pi)

    # --- quantize to HxW ---
    s_theta = math.pi / h
    s_phi   = 2 * math.pi / w
    itheta = torch.clamp((theta_bn_g1 / s_theta).floor().long(), 0, h - 1)
    iphi   = torch.clamp((phi_bn_g1   / s_phi).floor().long(),   0, w - 1)

    # --- compute h factor per anchor (Eq. 7) ---
    rho_max = rho_bn_g1.max(dim=1, keepdim=True).values    # (B*N, 1, 1)
    h_factor = compute_h(rho_bn_g1, rho_max)               # (B*N, G, 1)

    # --- build per-neighbor feature vector: [dx, dy, dz, h] ---
    feat_g4 = torch.cat([dp_rot, h_factor], dim=-1)        # (B*N, G, 4)

    # --- rasterize / accumulate ---
    D = feat_g4.shape[-1]
    freq = torch.zeros(B * N, h, w, D, device=device)
    # flatten batch*neighbors to single list of indices
    base = torch.arange(B * N, device=device).repeat_interleave(G)  # (B*N*G,)
    it  = itheta.view(-1)                                           # (B*N*G,)
    ip  = iphi.view(-1)                                             # (B*N*G,)
    vals= feat_g4.view(B * N * G, D)

    freq.index_put_((base, it, ip), vals, accumulate=True)

    # also return uv (for later gather back)
    uv = torch.stack([itheta.view(-1), iphi.view(-1)], dim=-1)      # (B*N*G, 2)
    return freq, uv, h_factor.view(-1, 1)



class ZEmb(nn.Module):
    """
    Paper-accurate surface-aware geometric encoder:
    - deterministic reference axis via farthest neighbor
    - spherical projection + quantization
    - rolling convolution (phi circular, theta zero)
    """
    def __init__(self, dim=1, h=6, w=6, k=32, kernel_h=5, kernel_w=5):
        super(ZEmb, self).__init__()
        self.h = h
        self.w = w
        self.k = k
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        # D = 4: [dx, dy, dz, h]  ->  out: 1
        self.net = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(kernel_h, kernel_w), stride=1, padding=(0, 0))
        )

    def forward(self, dp):
        """
        dp: (B, 3, N, G) relative neighbor coords
        returns: z embedding weights (B, 1, N, G)
        """
        B, _, N, G = dp.shape
        device = dp.device

        # 1) construct raster (B*N, H, W, D) and indices
        freq, uv, _ = _3d22d_constructor(dp, h=self.h, w=self.w)

        # 2) rolling pad: zero in theta (H), circular in phi (W)
        freq = freq.permute(0, 3, 1, 2).contiguous()   # (B*N, D, H, W)
        freq = roll_pad(freq, kernel_h=self.kernel_h, kernel_w=self.kernel_w)

        # 3) 2D conv
        freq_conv = self.net(freq).squeeze(1)          # (B*N, H, W)

        # 4) gather back per-neighbor value at its (theta, phi) bin
        BN = B * N
        H, W = self.h, self.w
        # 注意：卷积后输出 spatial 大小仍是 (H, W)（因为我们用的是 rolling pad）
        it = uv[:, 0].clamp(0, H - 1)
        ip = uv[:, 1].clamp(0, W - 1)
        out = freq_conv[torch.arange(BN, device=device).repeat_interleave(G), it, ip]  # (B*N*G,)
        out = out.view(B, N, G).unsqueeze(1)  # (B, 1, N, G)
        return out

class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 4
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                #  act_args=None if i == len(channels) - 2
                                                #                 and (self.use_res or is_head) else act_args,
                                                 act_args=act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)
            self.Zemb = ZEmb(dim=1)
            self.bn2d = nn.BatchNorm2d(channels2[-1], affine=False)

            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        if self.is_head:
            f = self.convs1(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # preconv

            f = self.convs1(f)
            if not self.all_aggr:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
            # grouping
            dp, fj = self.grouper(new_p, p, f)
            # conv on neighborhood_dp
            z = self.Zemb(dp)

            dp = torch.cat([dp, z], dim=1)
            pe = self.convs2(dp)
            # pe + fj
            f = pe + self.bn2d(fj-fi.unsqueeze(-1))
            b, _, n, k = fj.shape
            if fi is not None:
                f = self.pool(f)
            else:
                f = self.pool(f)
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f, pe


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,#2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        identity = f
        f = self.convs([p, f], pe)
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f, pe]


@MODELS.register_module()
class PointSENEncoder(nn.Module):
    r"""The Encoder for PointNext
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S.
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})
        self.act_args = kwargs.get('act_args', {'act': 'relu'})
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        Zemb = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append([])
                Zemb.append(nn.ModuleList())
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1
                ))
                Zemb.append(ZEmb(dim=1))
                pe_grouper.append(create_grouper(group_args))
        self.Zemb = nn.Sequential(*Zemb)
        self.encoder = nn.Sequential(*encoder)
        self.pe_encoder = pe_encoder #nn.Sequential(*pe_encoder)
        self.pe_grouper = pe_grouper
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        ## for PE of this stage
        channels2 = [4, channels]
        convs2 = []
        if blocks > 1:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                norm_args=self.norm_args,
                                                act_args=self.act_args,
                                                **self.conv_args)
                            )
            convs2 = nn.Sequential(*convs2)
            return convs2
        else:
            return nn.ModuleList()

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            pe = None
            p0, f0, pe = self.encoder[i]([p0, f0, pe])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            if i == 0:
                pe = None
                _p, _f, _ = self.encoder[i]([p[-1], f[-1], pe])
            else:
                _p, _f, _ = self.encoder[i][0]([p[-1], f[-1], pe])
                if self.blocks[i] > 1:
                    # grouping
                    dp, _ = self.pe_grouper[i](_p, _p, None)
                    # conv on neighborhood_dp
                    z = self.Zemb[i](dp)
                    dp = torch.cat([dp, z], dim=1)
                    pe = self.pe_encoder[i](dp)
                    _p, _f, _ = self.encoder[i][1:]([_p, _f, pe])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)



if __name__ == '__main__':
    # from torchsummaryX import summary
    # summary(reducenet56(10), inputsize=(3, 32, 32))
    from thop import profile
    from thop import clever_format
    x = torch.zeros(1, 3, 224, 224)

    flops, params = profile(net(), inputs=(x))
    # print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)