import os
os.environ["DEEPCHEM_DISABLE_MOLNET"] = "1"
os.environ["DEEPCHEM_DATA_DIR"] = "/data/deepchem_data"
import pubchempy as pcp
import torch
import numpy as np
import pandas as pd

# os.environ["DEEPCHEM_DATA_DIR"] = "/data/yanmeifang/deepchem_data"
#import deepchem as dc
#from deepchem.feat import CircularFingerprint, WeaveFeaturizer
import deepchem
deepchem.molnet = None
from deepchem.feat.molecule_featurizers import CircularFingerprint
from deepchem.feat import CircularFingerprint
from deepchem.feat.graph_features import WeaveFeaturizer
#from torch_scatter import scatter
import csv

# 读取文件
file_path = '/data/Classification/Kinase/CDK4-cyclinD3/CDK4-cyclinD3.csv'  # 请将此路径改为您的文件路径
drug_df = pd.read_csv(file_path)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 打开 CSV 文件，准备逐行写入 1D 和 2D 特征
features_1d_path = '/data/Classification/Kinase/CDK4-cyclinD3/drug_1d_fingerprints.csv'
features_2d_path = '/data/Classification/Kinase/CDK4-cyclinD3//drug_2d.csv'

# 创建并写入 CSV 文件的表头（1D特征的长度是固定的2048）
with open(features_1d_path, mode='w', newline='') as f1d, open(features_2d_path, mode='w', newline='') as f2d:
    writer_1d = csv.writer(f1d)
    writer_2d = csv.writer(f2d)
    
    # 写入 1D 特征的表头
   # writer_1d.writerow(['Drug_Name'] + [f'1D_Feature_{i}' for i in range(2048)])
    
    # 2D 特征没有固定列数，不预定义表头，后面每次根据特征长度动态写入

# 处理每个 CID，提取特征并逐行写入文件
for idx, row in drug_df.iterrows():
    cid = row['index']
    smiles = row['smiles']
    
    try:
        # 获取药物的 SMILES 表示
        # compound = pcp.Compound.from_cid(cid)
        # smiles = compound.canonical_smiles
        
        # 1D fingerprint 特征提取
        #featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        featurizer = CircularFingerprint(size=2048, radius=4)
        mols = [smiles]
        features_1d = featurizer.featurize(mols)
        drug_1d_features = torch.tensor(features_1d, dtype=torch.float32).to(device).flatten().cpu().numpy()
        #drug_1d_features = torch.nn.AdaptiveAvgPool1d(128)(torch.from_numpy(drug_1d_features).unsqueeze(0)).squeeze().numpy()
        # 2D graph 特征提取
        #featurizer_2d = dc.feat.WeaveFeaturizer()
        featurizer_2d = WeaveFeaturizer()
        features_2d = featurizer_2d.featurize([smiles])
        # weave_mol = features_2d[0] if isinstance(features_2d[0], dc.feat.graph_data.GraphData) else None
        # if weave_mol is not None and weave_mol.num_nodes > 1:
        #     atom_features = weave_mol.node_features
        #     pair_features = weave_mol.edge_features
        # else:
        #     # 用零向量填充
        #     atom_features = np.zeros((1, features_2d.num_atom_features))
        #     pair_features = np.zeros((1, features_2d.num_bond_features))
        atom_features = features_2d[0].get_atom_features()
        pair_features = features_2d[0].get_pair_features()
        atom_features_np = np.array(atom_features, dtype=np.float32)
        pair_features_np = np.array(pair_features, dtype=np.float32)

        atom_features_flat = atom_features_np.flatten()
        pair_features_flat = pair_features_np.flatten()

        combined_features_flat = np.concatenate([atom_features_flat, pair_features_flat])
        combined_features_flat = torch.from_numpy(combined_features_flat)
        combined_features_flat = combined_features_flat.unsqueeze(0).unsqueeze(0)
        #combined_features_flat = torch.nn.AdaptiveAvgPool1d(128)(torch.from_numpy(combined_features_flat).unsqueeze(0)).squeeze().numpy()
        combined_features_flat = torch.nn.functional.adaptive_avg_pool1d(combined_features_flat, 128)
        combined_features_flat = combined_features_flat.squeeze(0).squeeze(0)
        combined_features_flat_np = combined_features_flat.numpy()
        # 逐行写入 1D 和 2D 特征到文件
        with open(features_1d_path, mode='a', newline='') as f1d, open(features_2d_path, mode='a', newline='') as f2d:
            writer_1d = csv.writer(f1d)
            writer_2d = csv.writer(f2d)

            # 写入 1D 特征
            writer_1d.writerow([cid] + drug_1d_features.tolist())

            # 写入 2D 特征（每次写入时根据特征长度动态扩展列数）
            writer_2d.writerow([cid] + combined_features_flat.tolist())
    
    except Exception as e:
        print(f"Error processing CID {cid} ({cid}): {e}")
