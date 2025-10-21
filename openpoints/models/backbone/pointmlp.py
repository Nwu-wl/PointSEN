"""PointMLP

Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework
Xu Ma and Can Qin and Haoxuan You and Haoxi Ran and Yun Fu

Reference:
https://github.com/ma-xu/pointMLP-pytorch
"""
import string
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import furthest_point_sample, random_sample, LocalAggregation, create_convblock2d, three_interpolate, \
    three_nn, gather_operation, create_linearblock, create_convblock1d, create_grouper
import logging
import copy
from ..build import MODELS
from ..layers import furthest_point_sample, fps
from ..layers.group import QueryAndGroup
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import time



def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)



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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


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


def group_points(x, patch_num):
    B, N, C = x.shape
    x = x.view(B, patch_num, -1, C).reshape(B, patch_num, -1, 3)
    #  print(x.shape)
    return x


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, ):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


def weightReLu(x, active=nn.ReLU(inplace=True)):
    x = active(0.5 - x / (2 * x.max(-1,keepdim=True)[0])) + 0.5
    return x


def _3d22d_constructor_weight_pooling(xyz, new_xyz, feature=None, k=32, h=36, w=36, r=None, emb=None, softmax=None,
                                      bn1d=None):
    idx = query_ball_point(r, k, xyz, new_xyz) if r != None else knn_point(k, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
    group_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if feature is not None:
        # feature = index_points(feature, idx)
        feature = index_points(feature, idx)
    B, N, G, _ = group_xyz_norm.shape
    group_xyz_norm = group_xyz_norm.reshape(B * N, G, 3)
    sphere = xyz2sphere(group_xyz_norm)
    weight = weightReLu(sphere[:, :, 0]).unsqueeze(-1)
    if feature is not None:
        grouped_feature = feature
        sphere_xyz_feature = torch.cat([weight], dim=-1).reshape(B * N * G, -1)
    else:
        sphere_xyz_feature = torch.cat([weight], dim=-1).reshape(B * N * G, -1)
        grouped_feature = torch.cat([sphere, group_xyz_norm], dim=-1).reshape(B, N, G, 6)
    _2d_dim = sphere_xyz_feature.shape[-1]
    frequ_0 = torch.zeros(1).cuda().repeat(B * N, h, w, _2d_dim)

    sphere = sphere.reshape(-1, 3)
    u_v = torch.zeros(1, 2).cuda().repeat(B * N * G, 1)
    u_v[:, 0] = sphere[:, 1] * (h - 1)
    u_v[:, 1] = sphere[:, 2] * (w - 1)

    u_v = u_v.floor().type(torch.LongTensor).reshape(B * N * G, 2)

    frequ_0[np.arange(0, B * N, 1).repeat(G, axis=0).reshape(-1), u_v[:, 0], u_v[:, 1]] += sphere_xyz_feature[:,
                                                                                           0].unsqueeze(-1)

    feat = frequ_0.reshape(B, N, h, w, -1)
    weight = feat[:, :, :, :, 0]

    return weight,  grouped_feature, u_v, sphere[:, 0].reshape(B, N, G)


def fill_weight(x, kernel_x=7, kernel_y=7):
    x = torch.cat([x[:, :, :, -kernel_y // 2 + 1:], x, x[:, :, :, :kernel_y // 2]], dim=-1)
    # x = torch.cat([x[:, :, -kernel_x // 2 + 1:, :], x, x[:, :, :kernel_x // 2, :]], dim=-2)
    return x


class WeightConv(nn.Module):
    def __init__(self, weight_dim=1, expand=4):
        super(WeightConv, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(weight_dim, weight_dim * 4, kernel_size=(3, 5), stride=1, padding=(1, 0)),
                                 nn.BatchNorm2d(weight_dim * 4),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(weight_dim*4, weight_dim * 4, kernel_size=(3, 5), stride=1, padding=(1, 2)),
                                 nn.BatchNorm2d(weight_dim * 4),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(weight_dim * 4, weight_dim, kernel_size=1, stride=1, ),
                                 nn.BatchNorm2d(weight_dim)
                                          )
    def forward(self, x):
        return self.net(x)


class WeightFeatureConv(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, expand=4, n=1024):
        super(WeightFeatureConv, self).__init__()
        self.inplanes = in_dim * expand
        self.out_dim = out_dim
        self.n = n
        # self.net = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=(3, 5), stride=2, padding=1),
        #     nn.Conv2d(in_dim, self.inplanes, kernel_size=(3, 5), stride=1, padding=(1, 3), bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(2, 3), stride=2, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d((2, 3))
        # )
        self.net = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(3, 5), stride=2, padding=1),
            nn.Conv2d(in_dim, self.inplanes, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(3, 6), stride=(2, 3), bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=(4, 5), stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.mlp = ConvBNReLU1D(self.inplanes, out_dim)
        #self.mlp2 = ConvBNReLURes1D(out_dim)

    def forward(self, x):
        return self.mlp(self.net(x).reshape(-1, self.n, self.inplanes).permute(0, 2, 1))


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=False):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias)
        self.transfer1 = ConvBNReLU1D(out_channels, out_channels, bias=bias)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=False, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        # new_points = grouped_points+new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)
        return new_xyz, new_points


class PointMLPSetAbstraction(nn.Module):
    def __init__(self, num_point, num_sample, in_channel, out_channel):
        super(PointMLPSetAbstraction, self).__init__()
        self.num_point = num_point
        self.num_sample = num_sample
        self.sample_and_group_normal = LocalGrouper(in_channel, num_point, num_sample)
        self.down_sampling = PreExtraction(out_channel, out_channel)

    def forward(self, xyz, points):  # input[B,N,Dim]
        new_xyz, new_points = self.sample_and_group_normal(xyz, points)
        new_points = self.down_sampling(new_points)  # [B, dim, N]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

class P2iBlock(nn.Module):
    def __init__(self, input_dim, output_dim, npoints, r=None, is_head=False, group_all=False):
        super(P2iBlock, self).__init__()
        self.normalize = False
        self.emb_dim = 16
        self.input_dim = input_dim
        self.npoints = npoints
        self.r = r
        self.h = 9
        self.w = 18
        self.k = 24
        self.reduce = 'sum'
        self.is_head = is_head
        self.group_all = group_all
        self.downsample = PointMLPSetAbstraction(npoints, self.k, input_dim, output_dim)
        if is_head:
            self.emb = ConvBNReLU1D(6, self.emb_dim)
            self.ab_emb = ConvBNReLU1D(4, output_dim)
        self.short = ConvBNReLU1D(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-2)
        # self.bn1d = nn.LayerNorm(24, elementwise_affine=False)
        self.bn1d = nn.BatchNorm1d(24, affine=False)
        weight_dim = 1
        self.weight_conv = WeightConv(weight_dim=weight_dim)
        self.weight_f = WeightConv(weight_dim=1)
        # self.gf_extra5 = PreExtraction(self.emb_dim*16, self.emb_dim*32)
        self.bn = nn.BatchNorm1d(output_dim)
        self.distance_normal = nn.LayerNorm(self.k)
        self.softmax = nn.Softmax(dim=-1)
        self.confuse = ConvBNReLU1D(output_dim, output_dim)
        self.act = nn.ReLU()
        self.f_conv2d = WeightFeatureConv(6, output_dim//2, n=npoints) if is_head else WeightFeatureConv(6,
                                                                                                      output_dim,
                                                                                                      n=npoints)
        self.gf_extra = PreExtraction(9, output_dim) if is_head else PreExtraction(
            output_dim,
            output_dim)
        self.pos_extra = PreExtraction(6, output_dim) if is_head else PreExtraction(6, output_dim // 2)
        self.mlp = ConvBNReLU1D(output_dim, output_dim)
        self.proj = PreExtraction(output_dim, output_dim) if self.reduce=='maxpooling' else ConvBNReLURes1D(output_dim)


    def forward(self, xyz, f=None):
        if not self.is_head:
            new_xyz, f = self.downsample(xyz, f)
            identity = f
            f = self.mlp(f.permute(0, 2, 1)).permute(0, 2, 1)
            batch_size, n, _ = new_xyz.size()
            weight, grouped_feature, u_v, dist = _3d22d_constructor_weight_pooling(new_xyz, new_xyz=new_xyz,
                                                                                         feature=f,
                                                                                         k=self.k,
                                                                                         h=self.h, w=self.w, r=self.r,
                                                                                         softmax=self.softmax,
                                                                                         bn1d=self.bn1d)
            weight = weight.reshape(batch_size * n, -1, self.h, self.w)
            weight = fill_weight(weight, kernel_y=5)
            weight_f = self.weight_f(weight).squeeze(1)
            weight_f = weight_f[
                np.arange(0, batch_size * n, 1).repeat(self.k, axis=0).reshape(-1), u_v[:, 0], u_v[:, 1]].reshape(
                batch_size, n, self.k)+weightReLu(dist)
            weight_f = self.softmax(self.distance_normal(weight_f))
            f = grouped_feature * weight_f.unsqueeze(-1)
            if self.reduce == 'sum':
                f = torch.sum(f, dim=-2).reshape(batch_size, n, -1)
                f = self.proj(f.permute(0, 2, 1)).permute(0, 2, 1)
            elif self.reduce == 'maxpooling':
                f = self.proj(f).permute(0, 2, 1)
            f = self.act(f + identity)
        else:
            f = self.ab_emb(f).permute(0, 2, 1)
#             identity = f.permute(0, 2, 1)
#             f = self.mlp(f).permute(0, 2, 1)
#             batch_size, n, _ = xyz.size()
#             weight, grouped_feature, u_v, dist = _3d22d_constructor_weight_pooling(xyz, new_xyz=xyz,
#                                                                                          feature=f,
#                                                                                          k=self.k,
#                                                                                          h=self.h, w=self.w, r=self.r,
#                                                                                          softmax=self.softmax,
#                                                                                          bn1d=self.bn1d)
#             weight = weight.reshape(batch_size * n, -1, self.h, self.w)
#             weight = fill_weight(weight, kernel_y=5)

#             weight_f = self.weight_f(weight).squeeze(1)
#             weight_f = weight_f[
#                 np.arange(0, batch_size * n, 1).repeat(self.k, axis=0).reshape(-1), u_v[:, 0], u_v[:, 1]].reshape(
#                 batch_size, n, self.k)+weightReLu(dist)
#             weight_f = self.softmax(self.distance_normal(weight_f))
#             f = grouped_feature * weight_f.unsqueeze(-1)
#             if self.reduce == 'sum':
#                 f = torch.sum(f, dim=-2).reshape(batch_size, n, -1)
#                 f = self.proj(f.permute(0, 2, 1)).permute(0, 2, 1)
#             elif self.reduce == 'maxpooling':
#                 f = self.proj(f).permute(0, 2, 1)
#             f = self.act(f + identity)
            new_xyz = xyz

        return new_xyz, f


@MODELS.register_module()
class PointMLPEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs):
        super(PointMLPEncoder, self).__init__()
        self.normalize = False
        self.emb_dim = 32
        r_expand = 1.5
        self.absolut_pos_emb1 = ConvBNReLU1D(3, self.emb_dim*2)
        self.absolut_pos_emb2 = ConvBNReLU1D(self.emb_dim, self.emb_dim * 2)
        #         self.conv1 = nn.Conv1d(3, self.begin_dim, kernel_size=1, bias=False)
        #         self.conv2 = nn.Conv1d(self.begin_dim, self.begin_dim, kernel_size=1, bias=False)
        #         self.bn1 = nn.BatchNorm1d(self.begin_dim)
        #         self.bn2 = nn.BatchNorm1d(self.begin_dim)
        self.emb = ConvBNReLU1D(6, self.emb_dim)
        self.confu = ConvBNReLU1D(self.emb_dim * 2, self.emb_dim * 2)
        self.block1 = P2iBlock(self.emb_dim, self.emb_dim * 2, 1024, None, is_head=True)
        self.block2 = P2iBlock(self.emb_dim * 2, self.emb_dim * 4, 512, None)
        self.block3 = P2iBlock(self.emb_dim * 4, self.emb_dim * 8, 256, None)
        self.block4 = P2iBlock(self.emb_dim * 8, self.emb_dim * 16, 128, None)
        self.block5 = P2iBlock(self.emb_dim * 16, self.emb_dim * 32, 64, None)
        self.block6 = P2iBlock(self.emb_dim * 32, self.emb_dim * 64, 32, None)
        self.h = 9
        self.w = 18
        self.patch_num = 8
        self.aggregation = 'maxpooling'
        self.gf_extra_all = ConvBNReLURes1D(self.emb_dim * 32)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 15))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x, f0=None):

        return self.forward_cls_feat(x, f0)

    def forward_cls_feat(self, p, x=None):
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        batch_size, _, _ = x.size()

        xyz = p
        xyz, f = self.block1(xyz, f=x)
        xyz, f = self.block2(xyz, f)
        xyz, f = self.block3(xyz, f)
        xyz, f = self.block4(xyz, f)
        xyz, f = self.block5(xyz, f)
        f = F.adaptive_max_pool1d(f.permute(0, 2, 1), 1).squeeze(dim=-1)
        # print(_2d_p.shape)
        out = self.fc(f)
        return out


@MODELS.register_module()
class PointMLP(PointMLPEncoder):
    def __init__(self, in_channels=3, num_classes=15, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], group_args=None, **kwargs):
        super().__init__(in_channels, embed_dim, groups, res_expansion, activation, bias, use_xyz,
                         normalize, dim_expansion, pre_blocks, pos_blocks, k_neighbors, reducers,
                         **kwargs
                         )
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, p, x=None):
        return self.forward_cls_feat(p, x)

    def forward_cls_feat(self, p, x=None):
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        batch_size, _, _ = x.size()

        xyz = p
        xyz, f = self.block1(xyz, f=x)
        xyz, f = self.block2(xyz, f)
        xyz, f = self.block3(xyz, f)
        xyz, f = self.block4(xyz, f)
        xyz, f = self.block5(xyz, f)
        # f = self.gf_extra_all(f.permute(0, 2, 1))
        f = F.adaptive_max_pool1d(f.permute(0, 2, 1), 1).squeeze(dim=-1)
        # print(_2d_p.shape)
        out = self.fc(f)
        return out


# -------- There is Point Mlp Original Model Config
def pointMLP(num_classes=40, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                           activation="relu", bias=False, use_xyz=False, normalize="anchor",
                           dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                           k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(num_classes=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                           activation="relu", bias=False, use_xyz=False, normalize="anchor",
                           dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                           k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)
