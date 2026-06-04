import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.utils import negative_sampling
from models.renderer import MVRenderer
from models.multi_view import MVAgregate
from models.Act import ACT
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
from config import device



class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()
        # 第一层卷积
        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)
        # 第二层卷积
        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        # 第三层卷积
        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
        self.nor = nn.BatchNorm1d(256)
        

    def forward(self, x):
        x = x.unsqueeze(1)  
        

        x = F.relu(self.conv1(x))  
        x = self.pool(x)  

        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)  
        
        x = F.relu(self.conv3(x))  
        x = self.pool(x)  


        #x = x.mean(dim=[1, 2])
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self.pool2(x)
        x = x.squeeze(1)
        return x


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
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):

        x1 = self.lin1(x1)
        x2 = self.lin1(x2)
        x3 = self.lin1(x3)
        x1 = self.lin2(x1)
        x2 = self.lin2(x2)
        x3 = self.lin2(x3)
        x1 = self.lin3(x1)
        x2 = self.lin3(x2)
        x3 = self.lin3(x3)
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
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        if x3.dim() == 1:
            x3 = x3.unsqueeze(0)
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





class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3 ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(32)
        self.nor2 = torch.nn.BatchNorm1d(16)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)
        
        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)
        x = self.nor(x)
        x = self.relu(x)
        #x = self.drop(x)
        #x2 = self.l2(x)
        x = self.l2(x)
        #x = self.drop(x)
        x = self.nor2(x)
        x = self.relu(x)
        embedding = x
        x = self.l3(x)


        return x,embedding
    

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
    def __init__(self, nb_views=12,mini_batch_size=32):#mini_batch_size=32
        super(Act_3D_Aggregator, self).__init__()
        self.mini_batch_size = mini_batch_size
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
        total_size = points.shape[0]
        outputs = []
        batch_size = points.shape[0]
        for i in range(0, total_size, self.mini_batch_size):
            # 1. 切片获取当前小批次数据
            batch_points = points[i: i + self.mini_batch_size]
            current_bs = batch_points.shape[0]
            #print(f"  Rendering batch {i // self.mini_batch_size + 1} / {(total_size + self.mini_batch_size - 1) // self.mini_batch_size} ...")
            # 2. Act 预测 & 渲染 (只处理当前小批次)
            azim, elev, dist = self.Act(batch_points, c_batch_size=current_bs)
            rendered_images, _ = self.renderer(None, batch_points, azim=azim, elev=elev, dist=dist)
            # rendered_images: [current_bs, V, 3, H, W]

            # 3. 格式转换
            x_3d = rendered_images.permute(0, 2, 1, 3, 4)

            # 4. 3D CNN 特征提取
            feat_map = self.encoder_3d(x_3d)

            # 5. Act 聚合机制
            pooled_view, _ = torch.max(feat_map, dim=2)

            # 6. 空间维度池化
            global_feat = pooled_view.mean(dim=[2, 3])

            # 7. 投影
            out_batch = self.proj(global_feat)

            # 收集结果
            outputs.append(out_batch)#torch.cat(outputs, dim=0)
            last_azim, last_elev, last_dist = azim, elev, dist
            # --- 修正点 2: 将所有批次结果拼接成 [Total_Size, 128] ---
        if len(outputs) > 0:
            full_encode_3d = torch.cat(outputs, dim=0)
        else:
            # 这里的维度需要根据你的 proj 层输出对齐，通常是 128
            full_encode_3d = torch.empty((0, 128), device=points.device)
        # --- 将所有批次结果拼回一个大张量 ---
        return full_encode_3d, last_azim, last_elev, last_dist

class ActMol(nn.Module):
    def __init__(self,nb_views=14):
        super(ActMol, self).__init__()
        self.Act_3d_extractor = Act_3D_Aggregator(nb_views=nb_views)
        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.sp = Directional3DProcessor(128,64)
        self.con = conv2d()
        self.lne = nn.Linear(768,128)
        self.mlp_pre = mlp_pre(64,32,16,1)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.gate = gate()
        self.aaa = aaaa()
        self.intermediate_feature = None
        self.intermediate_gradient = None
    def forward(self, batch,drug_1d_features,drug_2d_features,drug_points,return_features=False):
        #encode_3d = self.Act_3d_extractor(drug_points)
        encode_3d, azim, elev, dist = self.Act_3d_extractor(drug_points)
        # encode_3d = drug_3d_features
        # encode_3d = self.sp(encode_3d)
        # encode_3d = encode_3d.mean(dim=[2, 3, 4])
        # encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)
        if encode_3d.dim() == 2:
            encode_3d = encode_3d.unsqueeze(1)
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)
        if drug_2d_features.dim() == 2:
            drug_2d_features = drug_2d_features.unsqueeze(1)
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)
        if drug_1d_features.dim() == 2:
            drug_1d_features = drug_1d_features.unsqueeze(1)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)

        weights = self.aaa(drug_1d_features,drug_2d_features, encode_3d)
        # weights = self.gate(drug_1d_features, drug_2d_features, encode_3d)
        self.saved_x = weights
        w1 = weights[:,0].unsqueeze(1)
        w2 = weights[:,1].unsqueeze(1)
        w3 = weights[:,2].unsqueeze(1)
        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3

        drug_feature = torch.cat((drug_1d_features,drug_2d_features,drug_3d_features), dim=1)
        if return_features:
            return None, drug_feature

        drug_feature = self.lne(drug_feature)

        drug_feature = self.nor(drug_feature)
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature

        edge_index = batch.edge_index.to(device)

        x_dict = self.conv1(drug_feature, edge_index)
        x_dict= self.nor(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv2(x_dict, edge_index)
        x_dict = self.nor2(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv3(x_dict,  edge_index)
        x_dict = self.nor3(x_dict)
        x_dict = self.relu(x_dict)


        return x_dict,self.saved_x,azim, elev, dist


    def compute_loss(self, out, batch):
        edge_index = batch.edge_label_index
        labels = batch.edge_label


        src = edge_index[0]
        dst = edge_index[1]

        drug1 = out[src]  #
        drug2 = out[dst]


        mout = torch.cat([drug1, drug2], dim=1)


        scores, t= self.mlp_pre(mout)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index, t



    def test(self,output,label):


        positive_class_probs = F.sigmoid(output)

        positive_class_probs = positive_class_probs.detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)

        aupr = average_precision_score(targets, positive_class_probs)
        positive_class_probs = positive_class_probs.flatten()
        targets = targets.flatten()
        positive_class_probs = (positive_class_probs > 0.6).astype(int)
        accuracy = accuracy_score(targets, positive_class_probs)


        precision = precision_score(targets, positive_class_probs)
        recall = recall_score(targets, positive_class_probs)
        return auc,aupr,accuracy,precision,recall

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

class ActMol(nn.Module):
    def __init__(self,nb_views=12):#in_channels, hidden_channels, hidden_channels2, out_channels_gat,out_channels, global_dim, num_layers, heads, ff_dropout,
                 #attn_dropout, spatial_size, skip, dist_count_norm, conv_type, num_centroids, no_bn, norm_type
        super(ActMol, self).__init__()
        self.Act_3d_extractor = Act_3D_Aggregator(nb_views=nb_views)
        self.conv1 = SAGEConv(256, 128)
        # self.conv11 = SAGEConv(128, 64)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.conv4 = GCNConv(16, 8)
        self.con5 = GCNConv(32, 16)
        self.gat = GATConv(256, 64, heads=4, concat=True)
        self.gat2 = GATConv(256, 128, heads=4, concat=False)
        self.jk_linear = nn.Linear(64 + 32 + 16, 32)  # 输出维度可调
        self.Autoencoder = Autoencoder()
        self.Autoencoder2 = ConvAutoencoder()
        self.sp = Directional3DProcessor(128, 64)
        self.con = conv2d()
        self.lne = nn.Linear(768, 128)
        self.mlp_pre = mlp_pre(64, 32, 16, 1)
        self.mlp_pre2 = mlp_pre(32, 16, 8, 1)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.relu2 = torch.nn.ReLU()
        self.tahn = torch.nn.Tanh()
        self.aaa = aaaa()
        self.gate = GateHead()
        self.fc = torch.nn.Linear(2, 1)

        self.intermediate_feature = None  # 用于存储中间特征
        self.intermediate_gradient = None  # 用于存储梯度

    def forward(self, batch, drug_1d_features, drug_2d_features, drug_points):
        #encode_3d = self.Act_3d_extractor(drug_points)
        encode_3d, azim, elev, dist = self.Act_3d_extractor(drug_points)
        # encode_3d = drug_3d_features
        # encode_3d = self.sp(encode_3d)
        # encode_3d = encode_3d.mean(dim=[2, 3, 4])
        # encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)
        if encode_3d.dim() == 2:
            encode_3d = encode_3d.unsqueeze(1)
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256).squeeze(1)
        if drug_2d_features.dim() == 2:
            drug_2d_features = drug_2d_features.unsqueeze(1)
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256).squeeze(1)
        if drug_1d_features.dim() == 2:
            drug_1d_features = drug_1d_features.unsqueeze(1)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256).squeeze(1)
        # # 融合***************************
        # drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256)
        # drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256)

        # drug_feature = torch.cat((drug_1d_features,drug_2d_features),dim=1)

        weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
        self.saved_x = weights
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)
        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3
        drug_feature = torch.cat((drug_1d_features, drug_2d_features, drug_3d_features), dim=1)
        drug_feature = self.lne(drug_feature)
        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 128)
        # drug_feature = self.nor(drug_feature)
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature
        # drug_feature = torch.cat((drug_res , drug_feature),dim=1)

        # **************************************************

        # only 3d
        # drug_feature = encode_3d
        # drug_feature = F.adaptive_avg_pool1d(drug_3d_features, 256).squeeze(0)

        # only 1d
        # drug_feature = drug_1d_features
        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 256).squeeze(0)

        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 256).squeeze(0)
        edge_index = batch.edge_index.to(device)
        drug_feature = F.adaptive_avg_pool1d(drug_feature.unsqueeze(0), 256).squeeze(0)
        x_dict = self.conv1(drug_feature, edge_index)
        x_dict = self.nor(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv2(x_dict, edge_index)
        x_dict = self.nor2(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv3(x_dict, edge_index)
        x_dict = self.nor3(x_dict)
        x_dict = self.relu(x_dict)

        # x_dict = self.conv4(x_dict, edge_index)
        # x_dict = self.relu(x_dict)

        # x_dict = self.nor3(x_dict)
        # x_dict = self.gat(drug_feature, edge_index)
        # x_dict = self.relu2(x_dict)
        # x_dict = self.nor2(x_dict)
        # x_dict = self.gat2(x_dict, edge_index)
        # x_dict = self.relu2(x_dict)

        # print(decoder.shape)
        return x_dict, self.saved_x,azim, elev, dist

    def compute_loss(self, out, batch):
        edge_index = batch.edge_label_index
        labels = batch.edge_label

        # 直接使用全部样本（假设正负样本已平衡）
        src = edge_index[0]
        dst = edge_index[1]

        # print(out.shape)
        drug1 = out[src]  # shape [N, dim]
        drug2 = out[dst]  # shape [N, dim]

        # 拼接方式构造边特征
        mout = torch.cat([drug1, drug2], dim=1)  # shape [N, 2*dim]

        # 前向分类
        scores, t = self.mlp_pre(mout)
        # print(scores.shape)
        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index, t

    def compute_loss2(self, out, batch):
        # 获取正负边对
        edge_index = batch.edge_label_index  # [2, N]
        labels = batch.edge_label  # [N]

        # 拆分出 drug1, drug2 的节点嵌入
        drug1 = out[edge_index[0]]  # [N, D]
        drug2 = out[edge_index[1]]  # [N, D]

        # —— 用内积来计算 logits scores ——
        # scores[i] = drug1[i] · drug2[i]
        scores = (drug1 * drug2).sum(dim=1)  # [N]

        # 二分类的 BCE + logits
        loss = F.binary_cross_entropy_with_logits(
            scores,
            labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index

    def test(self, output, label):

        # predictions = F.softmax(output, dim=1)
        positive_class_probs = F.sigmoid(output)
        # positive_class_probs = predictions[:, 1]  # 提取正类的概率
        positive_class_probs = positive_class_probs.detach().cpu().numpy()
        targets = label.cpu().numpy()

        # 计算AUC
        auc = roc_auc_score(targets, positive_class_probs)

        # 计算AUPR
        aupr = average_precision_score(targets, positive_class_probs)
        positive_class_probs = positive_class_probs.flatten()
        targets = targets.flatten()
        # df = pd.DataFrame({'prob': positive_class_probs, 'label': targets})

        # df.to_csv('graph/out.csv', index=False)
        # 计算Accuracy
        positive_class_probs = (positive_class_probs > 0.5).astype(int)
        accuracy = accuracy_score(targets, positive_class_probs)

        # 计算Precision
        precision = precision_score(targets, positive_class_probs)
        recall = recall_score(targets, positive_class_probs)
        return auc, aupr, accuracy, precision, recall

    def casepre(self, out, node_to_idx):
        # out 是每个节点的特征，node_to_idx 是从节点名到索引的映射
        # 获取 DB00619 的索引
        db00619_idx = node_to_idx['DB00619']  # 使用 node_to_idx 获取 DB00619 的索引

        # 获取 DB00619 的特征
        db00619_feature = out[db00619_idx]  # 这是 DB00619 的特征向量

        # 计算 DB00619 与其他所有节点的关系
        scores = []
        node_names = []  # 用于保存与 DB00619 的关系的节点名称
        for idx, node_feature in enumerate(out):
            if idx != db00619_idx:
                # 可以通过拼接 DB00619 的特征与其他节点的特征，或者直接相减/相加
                combined_feature = torch.cat((db00619_feature, node_feature), dim=0)  # 拼接两个特征
                score = self.mlp_pre(combined_feature)  # 使用 MLP 计算得分
                scores.append(score)
                node_names.append(list(node_to_idx.keys())[list(node_to_idx.values()).index(idx)])  # 获取节点名称

        # 将所有得分拼接成一个 tensor
        scores = torch.cat(scores, dim=0)
        scores = torch.sigmoid(scores)
        # 将得分与对应的节点名称一起保存到 CSV 文件
        results_df = pd.DataFrame({
            'node_name': node_names,
            'score': scores.cpu().numpy()  # 转换为 NumPy 数组以便保存
        })

        results_df.to_csv('graph_DDI/db00619_relationship_scores.csv', index=False)

        return scores, results_df  # 返回得分和保存的 DataFrame
