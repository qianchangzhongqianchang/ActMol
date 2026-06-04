import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
import math
from config import device
from Act import ACT
from models.renderer import MVRenderer
#from models.multi_view import MVAgregate
import torchvision
from models.renderer import MVRenderer
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class RGB_3D_Encoder(nn.Module):
    def __init__(self):
        super(RGB_3D_Encoder, self).__init__()
        # 结构与 3dfeature.py 中的 ConvAutoencoder1.encoder 一致
        # 但将输入通道从 4 改为 3
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=(1,2,2), padding=1), # stride=(1,..) 保留了视角维度
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
            nn.Conv3d(64,128, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.encoder(x)

class ACT_3D_Aggregator(nn.Module):
    def __init__(self, nb_views=12):
        super(ACT_3D_Aggregator, self).__init__()

        # 1. ACT 动态视角 (nb_views 可以随意设置)
        self.ACT = ACT(nb_views=nb_views,
                         views_config="learned_circular",
                         shape_extractor="PointNet",
                         shape_features_size=512)

        # 2. 渲染器
        self.renderer = MVRenderer(nb_views=nb_views,
                                   image_size=128,
                                   pc_rendering=True,
                                   object_color="white",
                                   background_color="black")

        # 3. 3D 特征提取器 (使用上面定义的 3通道版本)
        self.encoder_3d = RGB_3D_Encoder()

        # 4. 投影层 (融合前的维度调整)
        self.proj = nn.Linear(128, 128)

    def forward(self, points):
        """
        Input: points [Batch, N, 3]
        Output: features [Batch, 128]
        """
        batch_size = points.shape[0]

        # A. ACT 预测 & 渲染
        azim, elev, dist = self.ACT(points, c_batch_size=batch_size)
        #azim, elev, dist, shape_features = self.ACT(points,c_batch_size=batch_size, return_features=True)
        rendered_images, _ = self.renderer(None, points, azim=azim, elev=elev, dist=dist)
        # rendered_images shape: [B, V, 3, H, W]

        # B. 格式转换 -> [B, 3, V, H, W]
        # 将 Views (V) 视为 3D CNN 的 Depth (D)
        x_3d = rendered_images.permute(0, 2, 1, 3, 4)

        # C. 3D CNN 特征提取
        # Encoder 的 stride 是 (1,2,2)，所以 Depth (View) 维度保持不变
        feat_map = self.encoder_3d(x_3d)
        # feat_map shape: [B, 128, V, H', W']

        # D. ACT 聚合机制 (关键步骤)
        # 原 ACT 使用 Max Pooling 聚合多视角特征
        # 这里我们在 Depth (dim=2) 维度上进行 Max Pooling
        # 含义：在所有视角中，提取最显著的特征
        pooled_view, _ = torch.max(feat_map, dim=2)
        # pooled_view shape: [B, 128, H', W']

        # E. 空间维度池化 (Global Average Pooling)
        # 将 (H', W') 聚合为一个向量
        global_feat = pooled_view.mean(dim=[2, 3])
        # global_feat shape: [B, 128]

        # F. 投影 (可选，保持维度匹配)
        out = self.proj(global_feat)

        return out

class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
        self.nor = nn.BatchNorm1d(128)
    def forward(self, x):

        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self.pool2(x)
        x = x.squeeze(1)
        x = self.nor(x)
        return x


class SliceAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(SliceAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, D, H, W], D=6 is number of slices
        B, C, D, H, W = x.shape

        # Split into 3 parts along D (assume D=6)
        x1 = x[:, :, 0:2, :, :]
        x2 = x[:, :, 2:4, :, :]
        x3 = x[:, :, 4:6, :, :]

        def process_part(part):
            B, C, D, H, W = part.shape
            out = part.view(B, C, D * H * W).transpose(1, 2)  # [B, N, C]
            out, _ = self.attn(out, out, out)
            return out.transpose(1, 2).view(B, C, D, H, W)

        x1 = process_part(x1)
        x2 = process_part(x2)
        x3 = process_part(x3)

        # Concatenate along D (depth)
        return torch.cat([x1, x2, x3], dim=2)  # [B, C, D=6, H, W]
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 64, 1, 64, 64)
            nn.ReLU(True)
        )

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Attention block after bottleneck
        self.attn_block = SliceAttentionBlock(embed_dim=128, num_heads=4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_conv(x)
        x = self.attn_block(x)  # 融合 attention 后的编码表示
        x_recon = self.decoder(x)
        return x, x_recon


class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):
        x1 = F.relu(self.lin1(x1))
        x2 = F.relu(self.lin1(x2))
        x3 = F.relu(self.lin1(x3))
        x1 = F.relu(self.lin2(x1))
        x2 = F.relu(self.lin2(x2))
        x3 = F.relu(self.lin2(x3))
        x1 = F.relu(self.lin3(x1))
        x2 = F.relu(self.lin3(x2))
        x3 = F.relu(self.lin3(x3))
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        x1 = self.sigmoid(x1)
        x2 = self.sigmoid(x2)
        x3 = self.sigmoid(x3)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.softmax(x)
        return x

class aaaa(nn.Module):
    def __init__(self,
                 init_bias_c: float = 1.0,
                 max_epoch: int = 300,
                 max_bias: float = 5.0):
        super(aaaa, self).__init__()
        # 1) 三路输入共享的 BatchNorm
        # self.bn_input = nn.BatchNorm1d(256)
        self.bn_input = nn.BatchNorm1d(256)

        # 2) 三层 MLP（shared）
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)

        # 3) 池化 + softmax
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)

        # 4) 可训练静态偏置
        self.static_bias = nn.Parameter(
            torch.tensor([0., 0., init_bias_c], dtype=torch.float)
        )

        # 5) 动态偏置相关超参
        self.max_epoch = max_epoch
        self.max_bias  = max_bias

    def forward(self, x1, x2, x3, epoch: int = None):
        # --- 统一归一化到相同分布 ---
        x1 = self.bn_input(x1)
        x2 = self.bn_input(x2)

        x3 = self.bn_input(x3)

        # --- MLP 分支 ---
        def branch(x):
            x = F.relu(self.lin1(x))
            x = F.relu(self.lin2(x))
            x = F.relu(self.lin3(x))
            return self.pool(x.unsqueeze(1)).squeeze(1)  # -> [B, 32]

        a = branch(x1)
        b = branch(x2)
        c = branch(x3)

        # --- 拼接 logits ---
        logits = torch.cat([a, b, c], dim=1)  # [B,3]

        # --- 计算动态偏置 ---
        if epoch is not None:
            bias_c = (epoch / float(self.max_epoch)) * self.max_bias
        else:
            bias_c = 0.0
        dynamic_bias = torch.tensor([0., 0., bias_c],
                                    device=logits.device,
                                    dtype=logits.dtype)

        # --- 应用静态 + 动态偏置，再做 softmax ---
        logits = logits + self.static_bias + dynamic_bias
        weights = self.softmax(logits)         # [B,3]

        return weights

class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3,num_hid4 ,num_out ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.l4 = torch.nn.Linear(num_hid3, num_hid4)
        self.classify = torch.nn.Linear(num_hid4, num_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(num_hid1)
        self.nor2 = torch.nn.BatchNorm1d(num_hid2)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)

        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):

        x = self.l1(x)

       # x = self.drop(x)
        x = self.l2(x)

        #x = self.drop(x)
        x = self.l3(x)


        return x

class Directional3DProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Directional3DProcessor, self).__init__()

        self.conv_fr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_bb = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_tl = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoded_3d):  # [B, 128, 6, 32, 32]
        fr = encoded_3d[:, :, 0:2]  # [B, 128, 2, 32, 32]
        bb = encoded_3d[:, :, 2:4]
        tl = encoded_3d[:, :, 4:6]

        fr_out = self.conv_fr(fr)
        bb_out = self.conv_bb(bb)
        tl_out = self.conv_tl(tl)


        combined = torch.cat([fr_out, bb_out, tl_out], dim=2)
        return combined

#only3d
class MolVisClassifier1(nn.Module):
    def __init__(self, nb_views=12, output_dim=1):
        super(MolVisClassifier1, self).__init__()

        # 1. 3D 特征提取器 (ACT)
        # 输入: [Batch, N, 3], 输出: [Batch, 128]
        self.ACT_3d_extractor = ACT_3D_Aggregator(nb_views=nb_views)

        # 2. ResNet 特征增强模块
        # 你的 Conv1dNetwork 输入输出都是 128 维，适合做残差连接
        self.resnet = Conv1dNetwork()

        # 3. 归一化层
        self.nor = torch.nn.BatchNorm1d(128)

        # 4. 分类器 MLP
        # 输入维度 128 (来自 3D 提取器)，输出 1 (二分类概率 logits)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, drug_points):
        # --- 1. 3D 特征提取 ---
        # drug_points shape: [Batch, 150, 3]
        x_3d = self.ACT_3d_extractor(drug_points)  # 输出 [Batch, 128]

        # --- 2. 残差增强 (ResNet) ---
        x_3d = self.nor(x_3d)  # 归一化
        x_res = self.resnet(x_3d)  # 经过 Conv1d 网络
        x = x_3d + x_res  # 残差连接 (原特征 + 增强特征)

        # --- 3. MLP 分类预测 ---
        out = self.classifier(x)
        return out

    def compute_loss(self, out, labels):
        # 二分类 BCE Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.squeeze(), labels.float()
        )
        return loss


# class MolVisClassifier(nn.Module):
#     def __init__(self, nb_views=12, output_dim=1):
#         super(MolVisClassifier, self).__init__()
#
#         # 1. 3D 特征提取器 (保持不变，这是你的核心 ACT)
#         self.ACT_3d_extractor = ACT_3D_Aggregator(nb_views=nb_views)
#
#         # 2. 维度投影与归一化
#         self.lne = nn.Linear(768, 128)  # 假设融合后维度是 256*3 -> 768，或者根据你的aaa模块调整
#         self.nor = torch.nn.BatchNorm1d(128)
#         self.resnet = Conv1dNetwork()  # 你的ResNet模块
#
#         # 3. 多模态融合模块 (保留你的 aaa 或者 gate 模块)
#         self.aaa = aaaa()
#         # 或者使用你原本的 gate: self.gate = gate()
#
#         # 4. 分类器 MLP (替代原本的 GNN 交互层)
#         # 输入维度 128 (经过ResNet后)，输出 1 (二分类概率 logits)
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             #nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_dim)
#         )
#
#     def forward(self, drug_1d_features, drug_2d_features, drug_points):
#         # --- 1. 3D 特征提取 ---
#         encode_3d = self.ACT_3d_extractor(drug_points)  # [Batch, 128]
#
#         # --- 2. 维度对齐 (对齐到 256 以输入到 aaa) ---
#         # 注意：这里需要根据你 aaa 模块的输入维度调整，假设是 256
#         if encode_3d.dim() == 2: encode_3d = encode_3d.unsqueeze(1)
#         encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)
#
#         if drug_2d_features.dim() == 2: drug_2d_features = drug_2d_features.unsqueeze(1)
#         drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)
#
#         if drug_1d_features.dim() == 2: drug_1d_features = drug_1d_features.unsqueeze(1)
#         drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)
#
#         # --- 3. 特征融合 (使用 aaa 或 gate) ---
#         weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
#         w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
#
#         # 加权融合
#         feat_fused = torch.cat([
#             drug_1d_features * w2,  # 注意对应关系
#             drug_2d_features * w1,
#             encode_3d * w3
#         ], dim=1)  # [Batch, 768]
#
#         # --- 4. 特征压缩与增强 ---
#         x = self.lne(feat_fused)  # [Batch, 128]
#         x_res = self.resnet(x)
#         x = x + x_res
#
#         # --- 5. MLP 分类预测 ---
#         out = self.classifier(x)
#         return out
#
#     def compute_loss(self, out, labels):
#         # 二分类 BCE Loss
#         loss = torch.nn.functional.binary_cross_entropy_with_logits(
#             out.squeeze(), labels.float()
#         )
#         return loss

class MolVisClassifier(nn.Module):
    def __init__(self, nb_views=12, output_dim=1):
        super(MolVisClassifier, self).__init__()

        # 1. 3D 特征提取器 (保持不变，这是你的核心 ACT)
        self.ACT_3d_extractor = ACT_3D_Aggregator(nb_views=nb_views)

        # 2. 维度投影与归一化
        self.lne = nn.Linear(768, 128)  # 假设融合后维度是 256*3 -> 768，或者根据你的aaa模块调整
        self.nor = torch.nn.BatchNorm1d(128)
        self.resnet = Conv1dNetwork()  # 你的ResNet模块

        # 3. 多模态融合模块 (保留你的 aaa 或者 gate 模块)
        self.aaa = aaaa()
        # 或者使用你原本的 gate: self.gate = gate()

        # 4. 分类器 MLP (替代原本的 GNN 交互层)
        # 输入维度 128 (经过ResNet后)，输出 1 (二分类概率 logits)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    # def forward(self, drug_1d_features, drug_2d_features, drug_points):
    #     # --- 1. 3D 特征提取 ---
    #     encode_3d = self.ACT_3d_extractor(drug_points)  # [Batch, 128]
    #
    #     # --- 2. 维度对齐 (对齐到 256 以输入到 aaa) ---
    #     # 注意：这里需要根据你 aaa 模块的输入维度调整，假设是 256
    #     if encode_3d.dim() == 2: encode_3d = encode_3d.unsqueeze(1)
    #     encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)
    #
    #     if drug_2d_features.dim() == 2: drug_2d_features = drug_2d_features.unsqueeze(1)
    #     drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)
    #
    #     if drug_1d_features.dim() == 2: drug_1d_features = drug_1d_features.unsqueeze(1)
    #     drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)
    #
    #     # --- 3. 特征融合 (使用 aaa 或 gate) ---
    #     weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
    #     w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
    #
    #     # 加权融合
    #     feat_fused = torch.cat([
    #         drug_1d_features * w2,  # 注意对应关系
    #         drug_2d_features * w1,
    #         encode_3d * w3
    #     ], dim=1)  # [Batch, 768]
    #
    #     # --- 4. 特征压缩与增强 ---
    #     x = self.lne(feat_fused)  # [Batch, 128]
    #     x_res = self.resnet(x)
    #     x = x + x_res
    #
    #     # --- 5. MLP 分类预测 ---
    #     out = self.classifier(x)
    #     return out

        # 修改 graph_model_auto.py 中的 MolVisClassifier 的 forward 函数

    def forward(self, drug_1d_features, drug_2d_features, drug_points,
                return_features=False):  # <--- 添加 return_features 参数
        # --- 1. 3D 特征提取 ---
        encode_3d = self.ACT_3d_extractor(drug_points)  # [Batch, 128]

        # --- 2. 维度对齐 (对齐到 256 以输入到 aaa) ---
        if encode_3d.dim() == 2: encode_3d = encode_3d.unsqueeze(1)
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)

        if drug_2d_features.dim() == 2: drug_2d_features = drug_2d_features.unsqueeze(1)
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)

        if drug_1d_features.dim() == 2: drug_1d_features = drug_1d_features.unsqueeze(1)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)

        # --- 3. 特征融合 (使用 aaa 或 gate) ---
        weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # 加权融合
        feat_fused = torch.cat([
            drug_1d_features * w2,
            drug_2d_features * w1,
            encode_3d * w3
        ], dim=1)  # [Batch, 768]

        # --- 4. 特征压缩与增强 ---
        x = self.lne(feat_fused)  # [Batch, 128]
        x_res = self.resnet(x)
        x = x + x_res  # <--- 这是最终分类前的高维特征 (128维)

        # --- 5. MLP 分类预测 ---
        out = self.classifier(x)

        # === 新增：如果需要特征进行可视化，则返回特征 ===
        if return_features:
            return out, x
        return out

    def compute_loss(self, out, labels):
        # 二分类 BCE Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.squeeze(), labels.float()
        )
        return loss

# class MolVisClassifier(nn.Module):
#     def __init__(self, nb_views=12, output_dim=1):
#         super(MolVisClassifier, self).__init__()
#
#         # 1. 3D 特征提取器 (保持不变，这是你的核心 ACT)
#         self.ACT_3d_extractor = ACT_3D_Aggregator(nb_views=nb_views)
#
#         # 2. 维度投影与归一化
#         self.lne = nn.Linear(768, 128)  # 假设融合后维度是 256*3 -> 768，或者根据你的aaa模块调整
#         self.nor = torch.nn.BatchNorm1d(128)
#         self.resnet = Conv1dNetwork()  # 你的ResNet模块
#
#         # 3. 多模态融合模块 (保留你的 aaa 或者 gate 模块)
#         self.aaa = aaaa()
#         # 或者使用你原本的 gate: self.gate = gate()
#
#         # 4. 分类器 MLP (替代原本的 GNN 交互层)
#         # 输入维度 128 (经过ResNet后)，输出 1 (二分类概率 logits)
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             #nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_dim)
#         )
#
#     def forward(self, drug_1d_features, drug_2d_features, drug_points):
#         # --- 1. 3D 特征提取 ---
#         #encode_3d = self.ACT_3d_extractor(drug_points)  # [Batch, 128]
#         encode_3d, azim, elev, dist = self.ACT_3d_extractor(drug_points)
#         # --- 2. 维度对齐 (对齐到 256 以输入到 aaa) ---
#         # 注意：这里需要根据你 aaa 模块的输入维度调整，假设是 256
#         if encode_3d.dim() == 2: encode_3d = encode_3d.unsqueeze(1)
#         encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)
#
#         if drug_2d_features.dim() == 2: drug_2d_features = drug_2d_features.unsqueeze(1)
#         drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)
#
#         if drug_1d_features.dim() == 2: drug_1d_features = drug_1d_features.unsqueeze(1)
#         drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)
#
#         # --- 3. 特征融合 (使用 aaa 或 gate) ---
#         weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
#         w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
#
#         # 加权融合
#         feat_fused = torch.cat([
#             drug_1d_features * w2,  # 注意对应关系
#             drug_2d_features * w1,
#             encode_3d * w3
#         ], dim=1)  # [Batch, 768]
#
#         # --- 4. 特征压缩与增强 ---
#         x = self.lne(feat_fused)  # [Batch, 128]
#         x_res = self.resnet(x)
#         x = x + x_res
#
#         # --- 5. MLP 分类预测 ---
#         out = self.classifier(x)
#         return out,azim, elev, dist
#
#     def compute_loss(self, out, labels):
#         # 二分类 BCE Loss
#         loss = torch.nn.functional.binary_cross_entropy_with_logits(
#             out.squeeze(), labels.float()
#         )
#         return loss