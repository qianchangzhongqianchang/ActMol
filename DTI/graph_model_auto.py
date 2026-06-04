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
from models.renderer import MVRenderer
from models.multi_view import MVAgregate
import torchvision
from models.Act import ACT
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


class Act_3D_Aggregator(nn.Module):
    def __init__(self, nb_views=12):
        super(Act_3D_Aggregator, self).__init__()

        # 1. Act 动态视角 (nb_views 可以随意设置)
        self.Act = Act(nb_views=nb_views,
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

        # A. Act 预测 & 渲染
        azim, elev, dist = self.Act(points, c_batch_size=batch_size)
        rendered_images, _ = self.renderer(None, points, azim=azim, elev=elev, dist=dist)
        # rendered_images shape: [B, V, 3, H, W]

        # B. 格式转换 -> [B, 3, V, H, W]
        # 将 Views (V) 视为 3D CNN 的 Depth (D)
        x_3d = rendered_images.permute(0, 2, 1, 3, 4)

        # C. 3D CNN 特征提取
        # Encoder 的 stride 是 (1,2,2)，所以 Depth (View) 维度保持不变
        feat_map = self.encoder_3d(x_3d)
        # feat_map shape: [B, 128, V, H', W']

        # D. Act 聚合机制 (关键步骤)
        # 原 Act 使用 Max Pooling 聚合多视角特征
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


class Autoencoder2(nn.Module):
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
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 64, 1, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 3, 8, 512, 512)
            nn.Sigmoid()  # For normalized pixel values between [0, 1]
        )
    def forward(self, x):
        x = self.encoder(x)

        encode = self.bottleneck(x)

        x = self.decoder(encode)

        return encode,x


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

    
class ActMol(nn.Module):
    def __init__(self,nb_views=12):
        super(ActMol, self).__init__()
        self.conv1 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(256, 128),
            ('drug', 'interacts', 'Protein'): SAGEConv(256, 128)
        },aggr='mean')
        self.conv2 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(128, 64),
            ('drug', 'interacts', 'Protein'): SAGEConv(128, 64)
        },aggr='mean')

        self.conv3 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(64, 32),
            ('drug', 'interacts', 'Protein'): SAGEConv(64, 32)
        },  aggr='mean')

        self.Act_3d_extractor = Act_3D_Aggregator(nb_views=nb_views)
        #self.Autoencoder = Autoencoder()

        #self.autp = Autoencoder()

        self.nor = torch.nn.BatchNorm1d(256)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(128)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.5)
        #self.sp = Directional3DProcessor(128,64)
        self.gate = gate()
        self.mlp_pre = mlp_pre(64,32,16,1,1,1)

    def forward(self, x_dict, edge_index_dict,drug_2d_features,drug_points):
        encode_3d = self.Act_3d_extractor(drug_points)
        #encode_3d = drug_3d_features
        # encode_3d = self.sp(encode_3d)
        # encode_3d = encode_3d.mean(dim=[2, 3, 4])
        # encode_3d = F.adaptive_avg_pool1d(encode_3d, 128)

        # 1. 显式增加一个 Channel 维度
        # [Batch, Features] -> [Batch, 1, Features]
        if drug_2d_features.dim() == 2:
            drug_2d_features = drug_2d_features.unsqueeze(1)

        # drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 128).squeeze(0)
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 128).squeeze(1)
        if x_dict['drug'].dim() == 2:
            x_dict['drug'] = x_dict['drug'].unsqueeze(1)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128).squeeze(1)
        drug_1d_features = x_dict['drug']

        x = self.gate(x_dict['drug'],drug_2d_features, encode_3d)
        # self.saved_x = x.detach().cpu().numpy()
        drug_2d_features = drug_2d_features * x[:,1].unsqueeze(1)
        drug_1d_features = drug_1d_features * x[:,0].unsqueeze(1)
        encode_3d = encode_3d * x[:,2].unsqueeze(1)

        x_dict['drug'] = torch.cat((drug_1d_features,drug_2d_features,encode_3d), dim=1)

        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'].unsqueeze(0), 128).squeeze(0)
        x_dict['drug'] = self.nor3(x_dict['drug'])

        drug_res = self.resnet(x_dict['drug'])


        x_dict['drug'] = drug_res + x_dict['drug']



        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'].unsqueeze(1), 256).squeeze(1)
        if x_dict['Protein'].dim() == 2:
            x_dict['Protein'] = x_dict['Protein'].unsqueeze(1)
        x_dict['Protein'] = F.adaptive_avg_pool1d(x_dict['Protein'], 256).squeeze(1)


        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}



        return x_dict


    def compute_loss(self,out, batch):

        # 获取边
        edge_index = batch[('drug', 'interacts', 'Protein')].edge_label_index
        # 标签 
        labels =  batch[('drug', 'interacts', 'Protein')].edge_label
        scoreout = []
        for d,m in zip(edge_index[0],edge_index[1]) :
            
            Protein_feature = out['Protein'][m]
            drug_feature = out['drug'][d]
            edge_feature = torch.cat((drug_feature, Protein_feature ), dim=0)


            scoreout.append(edge_feature)
        scoreout = torch.stack(scoreout)

        scoreout = self.mlp_pre(scoreout)

        edge = edge_index
    
        scores = scoreout.to(device)
        labels = labels.to(device)
        scores = scores.squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())


        total_loss =  loss 

        return total_loss,scores, labels,edge

    def test(self, output, label):
        positive_class_probs = torch.sigmoid(output).detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)
        aupr = average_precision_score(targets, positive_class_probs)

        # 将概率转换为二进制预测
        predicted = (positive_class_probs > 0.5).astype(int)

        # 计算其他指标
        accuracy = accuracy_score(targets, predicted)
        precision = precision_score(targets, predicted, zero_division=0)
        recall = recall_score(targets, predicted, zero_division=0)
        f1 = f1_score(targets, predicted, zero_division=0)
        return auc, aupr, accuracy, precision, recall, f1


#调参
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

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

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

class GateHead(nn.Module):
    def __init__(self, in_dim=256, negative_slope=0.2):
        super(GateHead, self).__init__()
        # Modality A 专属
        self.lin1_a = nn.Linear(in_dim, 128)
        self.lin2_a = nn.Linear(128,    64)
        self.lin3_a = nn.Linear(64,     32)
        # Modality B 专属
        self.lin1_b = nn.Linear(in_dim, 128)
        self.lin2_b = nn.Linear(128,    64)
        self.lin3_b = nn.Linear(64,     32)
        # Modality C 专属
        self.lin1_c = nn.Linear(in_dim, 128)
        self.lin2_c = nn.Linear(128,    64)
        self.lin3_c = nn.Linear(64,     32)
        self.input_norm = nn.LayerNorm(in_dim)
        # 换成 LeakyReLU
        self.act = nn.LeakyReLU(negative_slope)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, negative_slope):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming uniform 适配 LeakyReLU
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=negative_slope,
                    nonlinearity='leaky_relu'
                )
                nn.init.zeros_(m.bias)
    def forward(self, x1, x2, x3):
        eps = 1e-6
        x1 = (x1 - x1.mean(dim=1,keepdim=True)) / (x1.std(dim=1,keepdim=True) + eps)
        x2 = (x2 - x2.mean(dim=1,keepdim=True)) / (x2.std(dim=1,keepdim=True) + eps)
        x3 = (x3 - x3.mean(dim=1,keepdim=True)) / (x3.std(dim=1,keepdim=True) + eps)
        # --- 支路 A ---
        a = self.act(self.lin1_a(x1))
        a = self.act(self.lin2_a(a))
        a = self.act(self.lin3_a(a))
        a = self.pool(a.unsqueeze(1)).squeeze(1)

        # --- 支路 B ---
        b = self.act(self.lin1_b(x2))
        b = self.act(self.lin2_b(b))
        b = self.act(self.lin3_b(b))
        b = self.pool(b.unsqueeze(1)).squeeze(1)

        # --- 支路 C ---
        c = self.act(self.lin1_c(x3))
        c = self.act(self.lin2_c(c))
        c = self.act(self.lin3_c(c))
        c = self.pool(c.unsqueeze(1)).squeeze(1)

        # 拼接后 softmax
        logits = torch.cat([a, b, c], dim=1)  # [B, 3]
        weights = self.softmax(logits)
        return weights
class conv2d(nn.Module):
    def __init__(self):
        super(conv2d, self).__init__()
        self.conv1 = nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = self.conv3(x)

        x = self.pool3(x)


        return x


class ActMol(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels2, out_channels_gat,
                 out_channels, global_dim, num_layers, heads, ff_dropout,
                 attn_dropout, spatial_size, skip, dist_count_norm, conv_type, num_centroids, no_bn, norm_type,
                 nb_views=12):
        super(ActMol, self).__init__()
        self.Act_3d_extractor = Act_3D_Aggregator(nb_views=nb_views)

        # === 修改点 1: 将普通的 SAGEConv 替换为 HeteroConv ===
        # 这样才能正确处理 drug -> Protein 和 Protein -> drug 的双向消息传递
        self.conv1 = HeteroConv({
            ('drug', 'interacts', 'Protein'): SAGEConv(256, 128),
            ('Protein', 'interacts', 'drug'): SAGEConv(256, 128)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('drug', 'interacts', 'Protein'): SAGEConv(128, 64),
            ('Protein', 'interacts', 'drug'): SAGEConv(128, 64)
        }, aggr='mean')

        self.conv3 = HeteroConv({
            ('drug', 'interacts', 'Protein'): SAGEConv(64, 32),
            ('Protein', 'interacts', 'drug'): SAGEConv(64, 32)
        }, aggr='mean')
        # ====================================================

        self.lne = nn.Linear(768, 128)
        # 你的 mlp_pre 参数之前修正过，保持修正后的状态
        self.mlp_pre = mlp_pre(64, 32, 16, 1, 1, 1)
        self.mlp_pre2 = mlp_pre(32, 16, 8, 1, 1, 1)

        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.aaa = aaaa()

        # 存储中间结果用于后续分析
        self.saved_x = None

    def forward(self, batch, drug_1d_features, drug_2d_features, drug_points):
        # 1. 处理 3D 特征
        encode_3d = self.Act_3d_extractor(drug_points)

        # 2. 维度调整 (统一到 256 维以便融合，或根据 aaa 的输入要求)
        if encode_3d.dim() == 2:
            encode_3d = encode_3d.unsqueeze(1)
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)

        if drug_2d_features.dim() == 2:
            drug_2d_features = drug_2d_features.unsqueeze(1)
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)

        if drug_1d_features.dim() == 2:
            drug_1d_features = drug_1d_features.unsqueeze(1)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)

        # 3. 特征融合 (使用 aaa 模块)
        weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
        self.saved_x = weights
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)

        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3

        # 拼接后通过线性层和 ResNet
        drug_feature = torch.cat((drug_1d_features, drug_2d_features, drug_3d_features), dim=1)
        drug_feature = self.lne(drug_feature)  # 768 -> 128
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature

        # === 修改点 2: 准备 HeteroConv 的输入 ===
        # 将药物特征调整为 256 维 (conv1 的输入要求)
        drug_feature = F.adaptive_avg_pool1d(drug_feature.unsqueeze(0), 256).squeeze(0)

        # 获取并处理 Protein 特征 (必须也调整为 256 维)
        protein_feature = batch['Protein'].x
        if protein_feature.dim() == 2:
            protein_feature = protein_feature.unsqueeze(1)
        protein_feature = F.adaptive_avg_pool1d(protein_feature, 256).squeeze(1)

        # 构建特征字典 x_dict
        x_dict = {
            'drug': drug_feature,
            'Protein': protein_feature
        }

        # 获取边索引字典
        edge_index_dict = batch.edge_index_dict

        # === 修改点 3: 使用 HeteroConv 进行卷积 ===
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.relu(self.nor(x)) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.relu(self.nor2(x)) for key, x in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: self.relu(self.nor3(x)) for key, x in x_dict.items()}

        return x_dict, self.saved_x

    def compute_loss(self, out, batch):
        # 获取标签和边索引
        edge_label_index = batch[('drug', 'interacts', 'Protein')].edge_label_index
        labels = batch[('drug', 'interacts', 'Protein')].edge_label

        src = edge_label_index[0]  # drug 索引
        dst = edge_label_index[1]  # Protein 索引

        # === 修改点 4: 从字典中分别获取 drug 和 Protein 的嵌入 ===
        # out 是一个字典 {'drug': ..., 'Protein': ...}
        drug_embed = out['drug'][src]
        protein_embed = out['Protein'][dst]

        # 拼接构造边特征
        mout = torch.cat([drug_embed, protein_embed], dim=1)

        # 前向分类
        scores = self.mlp_pre(mout)
        t = scores

        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_label_index, t

    def test(self, output, label):
        # 复用之前的 test 函数，逻辑不变
        positive_class_probs = torch.sigmoid(output).detach().cpu().numpy()
        targets = label.cpu().numpy()
        auc_val = roc_auc_score(targets, positive_class_probs)
        aupr_val = average_precision_score(targets, positive_class_probs)
        predicted = (positive_class_probs > 0.5).astype(int)
        accuracy = accuracy_score(targets, predicted)
        precision = precision_score(targets, predicted, zero_division=0)
        recall = recall_score(targets, predicted, zero_division=0)
        return auc_val, aupr_val, accuracy, precision, recall