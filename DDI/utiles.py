import csv
import torch
import pandas as pd
import numpy as np
import os
import glob
import trimesh
from plyfile import PlyData

from config import device
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
def torch_center_and_normalize(points, p=2):
    """
    将点云归一化到单位球/立方体中 (复用自 custom_dataset.py)
    """
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    if m > 1e-6:
        points = points / m
    else:
        pass
    return points

def read_ply_robust(ply_path, nb_points=150):
    """
    健壮的 PLY 读取函数：尝试 trimesh，失败则回退到 plyfile
    """
    pts = None
    # A. 尝试 Trimesh
    try:
        mesh = trimesh.load(ply_path, process=False)
        if hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0:
            pts = np.asarray(mesh.vertices, dtype=np.float32)[:, :3]
        elif hasattr(mesh, 'points') and mesh.points is not None and len(mesh.points) > 0:
            pts = np.asarray(mesh.points, dtype=np.float32)[:, :3]
    except Exception:
        pass

    # B. 回退到 Plyfile
    if pts is None:
        try:
            plydata = PlyData.read(ply_path)
            v = plydata['vertex'].data
            names = v.dtype.names
            if all(k in names for k in ('x', 'y', 'z')):
                pts = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32)
            else:
                coords = [v[names[i]] for i in range(min(3, len(names)))]
                pts = np.vstack(coords).T.astype(np.float32)
        except Exception as e:
            print(f"Error reading {ply_path}: {e}")
            return torch.zeros((nb_points, 3), dtype=torch.float32)

    # # C. 采样/补全 (NumPy 操作)
    # n_pts = len(pts)
    # if n_pts > nb_points:
    #     choice = np.random.choice(n_pts, nb_points, replace=False)
    #     pts = pts[choice]
    # elif n_pts < nb_points:
    #     choice = np.random.choice(n_pts, nb_points - n_pts, replace=True)
    #     pts = np.concatenate([pts, pts[choice]], axis=0)
    #
    # # D. 转 Tensor 并归一化
    # points_tensor = torch.from_numpy(pts).float()
    # points_tensor = torch_center_and_normalize(points_tensor)
    # C. 降采样 (仅处理点数过多的情况)
    # 如果点数不足，这里先不动，等归一化后再补0
    n_pts = len(pts)
    if n_pts > nb_points:
        choice = np.random.choice(n_pts, nb_points, replace=False)
        pts = pts[choice]

    # D. 转 Tensor 并归一化
    # 注意：此时 points_tensor 的长度可能小于 nb_points
    points_tensor = torch.from_numpy(pts).float()

    # 必须先归一化，再补0。否则补的0会拉偏均值(centroid)
    points_tensor = torch_center_and_normalize(points_tensor)

    # E. 补 0 (Padding)
    current_n = points_tensor.shape[0]
    if current_n < nb_points:
        num_missing = nb_points - current_n
        # 创建全0 Tensor
        zeros = torch.zeros((num_missing, 3), dtype=torch.float32)
        # 拼接到尾部
        points_tensor = torch.cat([points_tensor, zeros], dim=0)

    return points_tensor

def load_ply_dict_to_ram(ply_folder, nb_points=150):
    """
    加载文件夹下所有 .ply 文件到内存字典
    Key: 文件名前缀 (Drug ID), Value: PointCloud Tensor [N, 3]
    """
    ply_data = {}
    if not os.path.exists(ply_folder):
        print(f"Warning: Folder {ply_folder} not found.")
        return ply_data

    files = glob.glob(os.path.join(ply_folder, "*.ply"))
    print(f"Pre-loading {len(files)} PLY files from {ply_folder} ...")

    for f in files:
        # 假设文件名格式为 "DrugID.ply" 或 "DrugID_output.ply"
        # 提取 ID：取第一个 '.' 或 '_' 之前的部分
        filename = os.path.basename(f)
        key_name = filename.split('.')[0].split('_')[0]

        # 读取并处理
        points = read_ply_robust(f, nb_points)
        ply_data[key_name] = points

    print("PLY loading finished.")
    return ply_data

def get_batch_drug_points(node_idxs, ply_data_dict, device,idx_to_node):
    """
    根据 graph 中的 node_idx 获取对应的点云 batch
    """
    batch_points = []

    for idx in node_idxs:
        # 使用你原有的 get_drug_name 获取 ID 字符串
        #drug_name = get_drug_name(str(int(idx)))
        idx_val = int(idx)
        if idx_val in idx_to_node:
            drug_name = idx_to_node[idx_val]
        else:
            print(f"Warning: Index {idx_val} not found in idx_to_node mapping.")
            drug_name = "Unknown"
        # 查表
        if drug_name in ply_data_dict:
            points = ply_data_dict[drug_name]
        else:
            # 缺失处理：全0
            # print(f"Missing PLY for {drug_name}")
            points = torch.zeros((2048, 3), dtype=torch.float32)

        batch_points.append(points)

    return torch.stack(batch_points).to(device)

def get_mirna_id(mirna_name1):  
    file_path = 'graph/data/unique_ncrna_miRBase.csv'
    with open(file_path, 'r') as file:  
        reader = csv.reader(file)
        next(reader)  
        for row in reader:  
            mirna_name, mirna_id = row  
            if mirna_name == mirna_name1:  
                return mirna_id  
            
def get_mirna_features(mirna_name1):  
    mirna_id = get_mirna_id(mirna_name1)   
    file_path = 'kmer_features.csv'  
    with open(file_path, 'r') as file:  
        for line in file:  
            parts = line.strip().split(',')  
            if parts and parts[0] == mirna_id:  

                numeric_features = [float(feat) for feat in parts[1:]]  

                features_tensor = torch.tensor(numeric_features, dtype=torch.float32)  
                return features_tensor  

def lode1d_to_gpu(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file,header=None)
    

    drug_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values}

    gpu_feature_dict = {}

    for drug, features in drug_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[drug] = feature_tensor
    print(f" Features 1d on GPU: {csv_file}")
    return gpu_feature_dict

def lode2d_to_gpu(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file,header=None)
    

    drug_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values}

    gpu_feature_dict = {}

    for drug, features in drug_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[drug] = feature_tensor
    print(f" Features 1d on GPU: {csv_file}")
    return gpu_feature_dict



def load_mirna_features(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file)
    mirna_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values} 
    gpu_feature_dict = {}

    for mirna_id, features in mirna_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[mirna_id] = feature_tensor
    print(f" Features mirna on GPU: {csv_file}")
    return gpu_feature_dict

def get_drug_id(drug_name):
    with open('graph/data/drug_list.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)   
        for row in reader:  
            drug_name1, drug_id = row  
            if drug_name1 == drug_name:  
                return drug_id


def get_drug_name(drug_idex):
    with open('graph/data/drug_mapping.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:  
            drug_name1, drug_id = row  
            if drug_idex == drug_id:  
                return drug_name1

def load_npy_to_gpu(npy_files, device):
    gpu_data = {}
    for file_path in npy_files:
        # Load the .npy file as a numpy array
        np_array = np.load(file_path)
        # Convert the numpy array to a torch tensor
        tensor = torch.from_numpy(np_array)
        # Move the tensor to the specified device (GPU)
        tensor = tensor.to(device).type(torch.float32)
        # Get the file name without path
        file_name = os.path.basename(file_path)
        # Add the tensor to the dictionary with the file name as the key
        gpu_data[file_name] = tensor
    return gpu_data

from einops import rearrange
def to_3d(x):
    return rearrange(x, '  c d h w -> (c d) h w ')

def get_drug_features(data,gpu_data_3d):
    drug_features_3d_tensors = []
    for idex in data:  
        b = get_drug_name(str(idex.item()))
        a = get_drug_id(b)   
        key = a+".npy"
        drug_features = gpu_data_3d.get(key)

        new_num_features = 8  # 新的特征图数量  
        additional_features = new_num_features - drug_features.size(1)  

        zero_padding = torch.zeros(drug_features.size(0),additional_features, drug_features.size(2), drug_features.size(3), dtype=torch.float32).to(device) 

        drug_3d_features = torch.cat((drug_features, zero_padding), dim=1).to(device) 
        expamdedd_drug_features = to_3d(drug_3d_features)
        drug_features_3d_tensors.append(expamdedd_drug_features) 

    stacked_3d_tensor = torch.stack(drug_features_3d_tensors, dim=0)  
    return stacked_3d_tensor

def get_drug_2d_features(data,gpu_2d):
    drug_features_2d_tensors = []
    for idex in data:  

        b = get_drug_name(str(idex.item()))

        drug_features = gpu_2d.get(b)
        # print(drug_features)
        drug_features_2d_tensors.append(drug_features)

    stacked_2d_tensor = torch.stack(drug_features_2d_tensors, dim=0)  

    return stacked_2d_tensor