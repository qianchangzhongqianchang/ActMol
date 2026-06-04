import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import unit_spherical_grid, batch_tensor
from .pointnet import *
from torch import nn
from .gnn import GINExtractor
from .fusion import AttentionFusion
from .smiles_vector import SmilesDescriptorExtractor


class CircularViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1] - 90.0
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class SphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12,canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class RandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_distance=2.2,  transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        return c_views_azim, c_views_elev, c_views_dist


class LearnedDirectViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist
        else:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist + adjutment_vector[2] * c_views_dist + 0.1


class LearnedCircularViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1]
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                                                       c_views_dist.size(), device=c_views_dist.device)

        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class LearnedSphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class LearnedRandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class ViewSelector(nn.Module):
    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0,):
        super().__init__()
        self.views_config = views_config
        self.nb_views = nb_views
        if self.views_config == "circular" or self.views_config == "custom" or (self.views_config == "spherical" and self.nb_views == 4):
            self.chosen_view_selector = CircularViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                             canonical_distance=canonical_distance, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "spherical":
            self.chosen_view_selector = SphericalViewSelector(nb_views=nb_views,canonical_distance=canonical_distance,transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "random":
            self.chosen_view_selector = RandomViewSelector(nb_views=nb_views, canonical_distance=canonical_distance, transform_distance=transform_distance)
        elif self.views_config == "learned_circular" or (self.views_config == "learned_spherical" and self.nb_views == 4):
            self.chosen_view_selector = LearnedCircularViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                               canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "learned_direct":
            self.chosen_view_selector = LearnedDirectViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                               canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.views_config == "learned_spherical":
            self.chosen_view_selector = LearnedSphericalViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                                  canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.views_config == "learned_random":
            self.chosen_view_selector = LearnedRandomViewSelector(nb_views=nb_views, canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)


    def forward(self, shape_features=None, c_batch_size=1):
        return self.chosen_view_selector(shape_features=shape_features, c_batch_size=c_batch_size)



class FeatureExtractor(nn.Module):
    def __init__(self,  shape_features_size, views_config, shape_extractor, screatch_feature_extractor=False,use_gnn=False, gnn_output_dim=256,use_smiles_vec=False, smiles_vec_dim=256):
        super().__init__()
        self.shape_features_size = shape_features_size
        self.use_gnn = use_gnn  # +++ 新增 +++
        self.use_smiles_vec = use_smiles_vec
        # self.features_type = features_type
        if views_config == "circular" or views_config == "random" or views_config == "spherical" or views_config == "custom":
            self.features_origin = "zeros"
        # elif setup["return_extracted_features"]:
        #     self.features_origin = "pre_extracted"
        else:
            self.features_origin = "points_features"
            # --- 修改 (fe_model -> pc_model) ---
            if shape_extractor == "PointNet":
                #self.pc_model = PointNet(40, alignment=True)
                self.pc_model = PointNet(40, alignment=False)
                self.pc_feature_dim = 1024  # PointNet 全局特征维度
                # self.pc_feature_dim = 512
                #self.pc_feature_dim = 256
            elif shape_extractor == "DGCNN":
                self.pc_model = SimpleDGCNN(40)
                # self.pc_feature_dim = 1024
                self.pc_feature_dim = 256# DGCNN 全局特征维度
            # --- 结束修改 ---
            # if shape_extractor == "PointNet":
            #     self.fe_model = PointNet(40, alignment=True)
            # elif shape_extractor == "DGCNN":
            #     self.fe_model = SimpleDGCNN(40)
            if not screatch_feature_extractor:
                print(shape_extractor)
                # load_point_ckpt(self.fe_model,  shape_extractor,
                #                 ckpt_dir='./checkpoint')
                # --- 修改 (fe_model -> pc_model) ---
                load_point_ckpt(self.pc_model, shape_extractor,
                                ckpt_dir='./ActMol/checkpoint')
            # self.features_order = {"logits": 0,
            #                        "post_max": 1, "transform_matrix": 2}
             # +++ [修改] 多模态逻辑块 +++
            if self.use_gnn or self.use_smiles_vec:
                self.features_origin = "multimodal"

                # 1. GNN 初始化
                if self.use_gnn:
                    print(f"启用GNN (GIN) 特征提取器 (输出: {gnn_output_dim})")
                    self.gnn_model = GINExtractor(output_dim=gnn_output_dim)
                else:
                    gnn_output_dim = 0  # 没启用设为0

                # 2. [新增] SMILES Descriptor 初始化
                if self.use_smiles_vec:
                    print(f"启用 SMILES Descriptor 分支 (输出: {smiles_vec_dim})")
                    self.smiles_vec_model = SmilesDescriptorExtractor(output_dim=smiles_vec_dim)
                else:
                    smiles_vec_dim = 0  # 没启用设为0

                # 3. [修改] 融合模块 (传入3个维度)
                self.fusion_module = AttentionFusion(
                    pc_feature_dim=self.pc_feature_dim,
                    gnn_feature_dim=gnn_output_dim,
                    smiles_feature_dim=smiles_vec_dim,  # <--- [新增] 传入描述符维度
                    output_dim=self.shape_features_size
                )
            else:
                # 如果不使用GNN，我们仍需将 1024 维投影到 shape_features_size
                #print(f"仅使用点云特征: 投影 {self.pc_feature_dim} -> {self.shape_features_size}")
                 # 假设 .blocks.MLP 存在
                from .blocks import MLP
                self.pc_projector = MLP([self.pc_feature_dim, self.shape_features_size], act='relu', norm=True)
                # +++ 结束新增 +++
    # def forward(self, extra_info=None, c_batch_size=1):
    #     if self.features_origin == "zeros":
    #         return torch.zeros((c_batch_size, self.shape_features_size))
    #     # elif self.features_origin == "pre_extracted":
    #     #     extra_info = Variable(extra_info)
    #     #     return extra_info.view(c_batch_size, self.shape_features_size)
    #     elif self.features_origin == "points_features":
    #         extra_info = extra_info.transpose(1, 2).to(
    #             next(self.fe_model.parameters()).device)
    #         features = self.fe_model(extra_info)
    #         # if self.features_type == "logits_trans":
    #         #     return torch.cat((features[0].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
    #         # elif self.features_type == "post_max_trans":
    #         #     return torch.cat((features[1].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
    #         # else:
    #         #     return features[self.features_order[self.features_type]].view(c_batch_size, -1)
    #         return features[0].view(c_batch_size, -1)
    def forward(self, extra_info=None, c_batch_size=1):

        if self.features_origin == "zeros":
            # 确保在
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if len(list(self.parameters())) > 0:
                device = next(self.parameters()).device
            return torch.zeros((c_batch_size, self.shape_features_size)).to(device)

        # # +++ 新增: 多模态路径 +++
        # elif self.features_origin == "multimodal":
        #     # extra_info 是一个元组 (points, smiles_list)
        #     points, smiles_list = extra_info
        #
        #     # 1. 提取点云特征
        #     points_input = points.transpose(1, 2).to(next(self.pc_model.parameters()).device)
        #     # ***重要***: 我们使用 [1] (global_feature), 而不是 [0] (logits)
        #     pc_features = self.pc_model(points_input)[1]
        #     pc_features = pc_features.view(c_batch_size, -1)  # [B, 1024]
        #     # .isfinite() 会同时捕获 NaN 和 Inf (无穷大)
        #     if not torch.all(torch.isfinite(pc_features)):
        #         print(f"!!!!!! [DEBUG] NaN/Inf DETECTED in pc_features (PointNet output) !!!!!!")
        #         # 打印这批SMILES，帮你定位是哪个分子出了问题
        #         print(f"       FOR SMILES BATCH: {smiles_list}")
        #     # 2. 提取GNN特征 (gnn_model 内部处理SMILES和GPU)
        #     gnn_features = self.gnn_model(smiles_list)  # [B, gnn_output_dim]
        #
        #     # # --- 这是你的检查 2 (建议使用 isfinite 并打印 smiles) ---
        #     # if not torch.all(torch.isfinite(gnn_features)):
        #     #     print(f"!!!!!! [DEBUG] NaN/Inf DETECTED in gnn_features (GNN output) !!!!!!")
        #     #     # 打印这批SMILES，帮你定位是哪个分子出了问题
        #     #     print(f"       FOR SMILES BATCH: {smiles_list}")
        #     # # 3. 融合特征
        #     fused_features = self.fusion_module(pc_features, gnn_features)  # [B, shape_features_size]
        #     return fused_features
        # +++ [修改] 多模态路径 +++
        elif self.features_origin == "multimodal":
            points, smiles_list = extra_info
            device = next(self.pc_model.parameters()).device

            # 1. 提取点云特征
            points_input = points.transpose(1, 2).to(device)
            pc_features = self.pc_model(points_input)[1].view(c_batch_size, -1)

            # 调试信息 (可选)
            if not torch.all(torch.isfinite(pc_features)):
                print(f"!!! NaN/Inf in PC features")

            # 2. 提取 GNN 特征
            if self.use_gnn:
                gnn_features = self.gnn_model(smiles_list)
            else:
                # 如果没启用，创建一个空张量，维度0为0
                gnn_features = torch.empty(c_batch_size, 0).to(device)

            # 3. [新增] 提取 SMILES Descriptor 特征
            if self.use_smiles_vec:
                smiles_vec_features = self.smiles_vec_model(smiles_list)
            else:
                # 如果没启用，创建一个空张量
                smiles_vec_features = torch.empty(c_batch_size, 0).to(device)

            # 4. [修改] 三路融合 (传入3个参数)
            fused_features = self.fusion_module(pc_features, gnn_features, smiles_vec_features)

            return fused_features

        # --- 修改: 原始 'points_features' 路径 ---
        elif self.features_origin == "points_features":
            # extra_info 现在只是 points
            points = extra_info
            points_input = points.transpose(1, 2).to(next(self.pc_model.parameters()).device)

            # ***重要***: 我们使用 [1] (global_feature)
            features = self.pc_model(points_input)
            pc_features = features[1].view(c_batch_size, -1)  # [B, 1024]
            # 强制检查，防止异常传递到渲染器
            if torch.isnan(pc_features).any():
                pc_features = torch.nan_to_num(pc_features, 0.0)  # 暂时用0填充防止死循环
            # 投影到目标维度
            projected_features = self.pc_projector(pc_features)
            return projected_features
        # --- 结束修改 ---


class ACT(nn.Module):


    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=2.2, transform_distance=False, input_view_noise=0.0, shape_extractor="pointnet", shape_features_size=512, screatch_feature_extractor=False,
                 use_gnn=False,gnn_output_dim=256,use_smiles_vec=False, smiles_vec_dim=256):
        super().__init__()
        self.view_selector = ViewSelector(nb_views=nb_views, views_config=views_config, canonical_elevation=canonical_elevation, canonical_distance=canonical_distance,
                                          shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise,)
        self.feature_extractor = FeatureExtractor(shape_features_size=shape_features_size, views_config=views_config,
                                                  shape_extractor=shape_extractor, screatch_feature_extractor=screatch_feature_extractor,
                                                  # +++ 新增 +++
                                                  use_gnn=use_gnn,
                                                  gnn_output_dim=gnn_output_dim,
                                                  use_smiles_vec=use_smiles_vec,  # <--- 传进去
                                                  smiles_vec_dim=smiles_vec_dim
                                                  )
        self.use_gnn = use_gnn  # +++ 新增 +++
        self.use_smiles_vec = use_smiles_vec

    def forward(self, points=None, c_batch_size=1,smiles=None,return_features=False):

        if self.use_gnn or self.use_smiles_vec:
            if smiles is None:
                raise ValueError("ACT配置为使用多模态(GNN或Descriptor)，但在forward()中未提供'smiles'参数。")
            extra_info = (points, smiles)
        else:
            extra_info = points
        # shape_features = self.feature_extractor(points, c_batch_size)
        # --- 修改: 传入 extra_info ---
        shape_features = self.feature_extractor(extra_info=extra_info,c_batch_size=c_batch_size)
        # 获取视角参数 (view_selector 只需要特征来决定视角)
        azim, elev, dist = self.view_selector(shape_features=shape_features, c_batch_size=c_batch_size)
        # --- 结束修改 ---
        # return self.view_selector(shape_features=shape_features, c_batch_size=c_batch_size)
        # <--- 2. 修改返回逻辑：根据 flag 决定是否返回特征
        if return_features:
            return azim, elev, dist, shape_features
        else:
            return azim, elev, dist


        #

    def load_act(self,weights_file):
        # Load checkpoint.
        print('\n==> Loading checkpoint..')
        assert os.path.isfile(weights_file
                            ), 'Error: no checkpoint file found!'
        checkpoint = torch.load(weights_file)
        self.load_state_dict(checkpoint['act'])
