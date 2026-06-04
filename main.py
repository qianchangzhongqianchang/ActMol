#from models.get_graph_auto import *
# from torch_geometric.loader import HGTLoader,neighbor_loader,NeighborLoader,LinkNeighborLoader
from torch_geometric.loader import HGTLoader, neighbor_loader
from graph_model_auto import MolVisClassifier1
from models.graph_model_auto import *
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import gc
from models.config import device
from models.utiles_auto import lode1d_to_gpu, load_Protein_features, get_drug_2d_features, load_ply_dict_to_ram, get_batch_drug_points
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 

print(f"Using device: {device}")  

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help="Random seed for model and dataset.")
parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default=0.00057786748871315845, help='learning rate in optimizer')#0.00057786748871315845 #0.0007097990120262436 0.00057786748871315845
parser.add_argument('--wd', type=float, default=1.5385723375708275e-06, help='weight decay in optimizer')#1.5385723375708275e-06 #0.00042202402410465296 1.5385723375708275e-06
parser.add_argument('--batch_size', type=int, default=32, help='Batch size') # 新增 batch_size 参数  #0.0037374676450013856 0.0009628440790918227
args = parser.parse_args()


best_val_loss = float('inf')

# #only3d
# class HIV3DDataset(Dataset):
#     def __init__(self, csv_path, ply_data_dict):
#         """
#         csv_path: 包含 index 和 label 的 csv 文件路径
#         ply_data_dict: 预加载到内存的 3D 点云字典
#         """
#         self.df = pd.read_csv(csv_path)
#         self.ply_data_dict = ply_data_dict
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         mol_idx = row['index']  # 确保 CSV 里有 'index' 这一列
#         label = row['label']
#
#         # 获取点云 (如果字典里找不到，返回全0，防止报错)
#         # 注意：这里假设 ply_data_dict 的 key 是字符串类型的 index
#         points = self.ply_data_dict.get(str(mol_idx))
#
#         if points is None:
#             # 尝试转 int 再转 str，防止因为类型不匹配（比如 '1.0' vs '1'）导致找不到
#             points = self.ply_data_dict.get(str(int(mol_idx)))
#
#         if points is None:
#             # print(f"Warning: Missing PLY for index {mol_idx}")
#             points = torch.zeros((150, 3), dtype=torch.float32)
#
#         return {
#             'points': points,
#             'label': torch.tensor(label, dtype=torch.float32)
#         }
#
#
# # ==========================================
# # 2. 指标计算函数
# # ==========================================
# def calculate_metrics(outputs, targets):
#     probs = torch.sigmoid(outputs).detach().cpu().numpy()
#     targets = targets.detach().cpu().numpy()
#     predicted = (probs > 0.5).astype(int)
#
#     try:
#         auc = roc_auc_score(targets, probs)
#     except ValueError:
#         auc = 0.0
#
#     aupr = average_precision_score(targets, probs)
#     accuracy = accuracy_score(targets, predicted)
#     precision = precision_score(targets, predicted, zero_division=0)
#     recall = recall_score(targets, predicted, zero_division=0)
#     f1 = f1_score(targets, predicted, zero_division=0)
#
#     return auc, aupr, accuracy, precision, recall, f1
#
#
# # ==========================================
# # 3. 评估函数
# # ==========================================
# def evaluate(loader, model, device):
#     model.eval()
#     all_scores = []
#     all_labels = []
#     total_loss = 0
#     criterion = torch.nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for batch in loader:
#             points = batch['points'].to(device)
#             labels = batch['label'].to(device)
#
#             # 前向传播 (只传入 points)
#             out = model(points).squeeze()
#
#             # 这里的 mask 用于处理 batch size = 1 的情况或者维度压缩问题
#             if out.ndim == 0: out = out.unsqueeze(0)
#
#             loss = criterion(out, labels)
#             total_loss += loss.item()
#
#             all_scores.append(out)
#             all_labels.append(labels)
#
#     if len(all_scores) == 0:
#         return 0, 0, 0, 0, 0, 0, 0
#
#     all_scores = torch.cat(all_scores)
#     all_labels = torch.cat(all_labels)
#
#     avg_loss = total_loss / len(loader)
#     auc, aupr, acc, prec, rec, f1 = calculate_metrics(all_scores, all_labels)
#
#     return avg_loss, auc, aupr, acc, prec, rec, f1
#
#
# def set_random_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# # ==========================================
# # 4. 主函数
# # ==========================================
# def main():
#     seed = args.seed
#     set_random_seed(seed)
#
#     # --- 路径设置 (请修改为实际路径) ---
#     # CSV 必须包含表头: index, label (index 对应 ply 文件名)
#     csv_file_path = './data/BBBP/bbbp.csv'
#     # PLY 文件夹路径
#     ply_folder_path = './data/BBBP/ply_output/'
#     model_save_path = 'model_path/bbbp_3D_Only_best.pth'
#
#     # 确保保存目录存在
#     os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
#
#     # --- 1. 加载 PLY 数据到内存 ---
#     print(f"Loading PLY data from {ply_folder_path}...")
#     # nb_points=150 需要与模型中的 input 维度或者 ACT 处理逻辑一致
#     ply_data_dict = load_ply_dict_to_ram(ply_folder_path, nb_points=150)
#     print(f"Loaded {len(ply_data_dict)} ply files.")
#
#     # --- 2. 构建 Dataset 和 DataLoader ---
#     full_dataset = HIV3DDataset(csv_file_path, ply_data_dict)
#
#     # 划分数据集 (80% 训练, 10% 验证, 10% 测试)
#     total_len = len(full_dataset)
#     train_size = int(0.8 * total_len)
#     val_size = int(0.1 * total_len)
#     test_size = total_len - train_size - val_size
#
#     train_data, val_data, test_data = torch.utils.data.random_split(
#         full_dataset, [train_size, val_size, test_size]
#     )
#
#     print(f"Data split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
#
#     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
#
#     # --- 3. 初始化模型 ---
#     model = MolVisClassifier1(nb_views=12, output_dim=1)
#     model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     criterion = torch.nn.BCEWithLogitsLoss()
#
#     best_auc = -float('inf')
#     best_model_state_dict = None
#     results_list = []
#
#     pbar = tqdm(range(args.epochs), desc="Training")
#
#     for epoch in pbar:
#         model.train()
#         train_loss = 0
#
#         for batch in train_loader:
#             points = batch['points'].to(device)
#             labels = batch['label'].to(device)
#
#             optimizer.zero_grad()
#
#             # 前向传播 (只传入 points)
#             out = model(points).squeeze()
#
#             # 处理 batch_size=1 导致的维度问题
#             if out.ndim == 0: out = out.unsqueeze(0)
#
#             loss = criterion(out, labels)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#
#         # 每 10 epoch 验证一次
#         if epoch % 10 == 0:
#             val_loss, auc, aupr, acc, prec, rec, f1 = evaluate(val_loader, model, device)
#
#             pbar.set_postfix({'Val AUC': f'{auc:.4f}', 'Loss': f'{train_loss / len(train_loader):.4f}'})
#
#             # 保存最佳模型
#             if auc > best_auc:
#                 best_auc = auc
#                 best_model_state_dict = model.state_dict()
#                 torch.save(best_model_state_dict, model_save_path)
#
#             # 记录日志
#             results_list.append([epoch, val_loss, auc, aupr, acc, prec, rec, f1])
#             print(f'\n| val_auc:{auc:.4f} | val_aupr:{aupr:.4f}| val_acc:{acc:.4f} | val_prec:{prec:.4f}| val_rec:{rec:.4f}| val_f1:{f1:.4f}')
#
#     # --- 4. 最终测试 ---
#     print("\nTraining Finished. Testing best model...")
#     if best_model_state_dict is not None:
#         model.load_state_dict(best_model_state_dict)
#
#     test_loss, t_auc, t_aupr, t_acc, t_prec, t_rec, t_f1 = evaluate(test_loader, model, device)
#
#     print("=" * 30)
#     print(f"Test Results:\nAUC: {t_auc:.4f}\nAUPR: {t_aupr:.4f}\nAccuracy: {t_acc:.4f}\nPrecision: {t_prec:.4f}\nRecall: {t_rec:.4f}\nF1: {t_f1:.4f}")
#     print("=" * 30)
#
#
# if __name__ == '__main__':
#     main()


class HIVDataset(Dataset):
    def __init__(self, csv_path, gpu_1d, gpu_2d, ply_data_dict):
        """
        csv_path: hiv.csv 路径
        gpu_1d: 1D 特征字典 (在显存或内存中)
        gpu_2d: 2D 特征字典 (在显存或内存中)
        ply_data_dict: 3D 点云字典 (在内存中)
        """
        self.df = pd.read_csv(csv_path)
        self.gpu_1d = gpu_1d
        self.gpu_2d = gpu_2d
        self.ply_data_dict = ply_data_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 假设 csv 中有一列 'index' 对应特征文件名 (如 1 -> 1.ply)
        # 如果你的 csv 是用 smiles 做索引，这里需要修改为 key = row['smiles']
        mol_idx = row['index']
        label = row['label']

        # 获取特征 (如果找不到则返回全0)
        # 注意：这里直接获取 Tensor。如果 gpu_1d 已经在 GPU 上，num_workers 必须设为 0
        feat_1d = self.gpu_1d.get(int(mol_idx))
        if feat_1d is None: feat_1d = torch.zeros(128, device=device)  # 维度需根据实际调整

        feat_2d = self.gpu_2d.get(int(mol_idx))
        if feat_2d is None: feat_2d = torch.zeros(128, device=device)  # 维度需根据实际调整

        # 获取点云 (通常在 CPU 内存中)
        points = self.ply_data_dict.get(str(mol_idx))
        if points is None: points = torch.zeros((150, 3), dtype=torch.float32)

        return {
            '1d': feat_1d,
            '2d': feat_2d,
            'points': points,
            'label': torch.tensor(label, dtype=torch.float32),
            'mol_name': str(mol_idx)  # 使用 indices 中的索引作为分子名称
        }


# ==========================================
# 2. 指标计算函数 (替代原模型中的 test 方法)
# ==========================================
def calculate_metrics(outputs, targets):
    # outputs: Logits (未经过 sigmoid)
    # targets: 真实标签 (0 或 1)

    # 转换为概率
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # 预测类别 (阈值 0.5)
    predicted = (probs > 0.5).astype(int)

    # 计算指标
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0  # 处理只有一个类别的情况

    aupr = average_precision_score(targets, probs)
    accuracy = accuracy_score(targets, predicted)
    precision = precision_score(targets, predicted, zero_division=0)
    recall = recall_score(targets, predicted, zero_division=0)
    f1 = f1_score(targets, predicted, zero_division=0)

    return auc, aupr, accuracy, precision, recall, f1


# ==========================================
# 3. 验证/测试 循环函数
# ==========================================
def evaluate(loader, model, device):
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            # 数据移动到 GPU
            feat_1d = batch['1d'].to(device)
            feat_2d = batch['2d'].to(device)
            points = batch['points'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            out = model(feat_1d, feat_2d, points).squeeze()

            # 计算 Loss
            loss = criterion(out, labels)
            total_loss += loss.item()

            all_scores.append(out)
            all_labels.append(labels)

    # 拼接所有批次结果
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # 计算平均 Loss 和所有指标
    avg_loss = total_loss / len(loader)
    auc, aupr, acc, prec, rec, f1 = calculate_metrics(all_scores, all_labels)

    return avg_loss, auc, aupr, acc, prec, rec, f1


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 4. 主函数
# ==========================================
def main():
    seed = args.seed
    set_random_seed(seed)

    # --- 路径设置 (请修改为你自己的实际路径) ---
    csv_file_path = './data/bace/bace.csv'  # 你的 excel文件保存为csv
    ply_folder_path = './data/bace/ply_output/'  # HIV的点云文件夹
    fingerprint_path = './data/bace/drug_1d_fingerprints.csv'
    drug_2d_path = './data/bace/drug_2d.csv'
    model_save_path = 'model_path/bace_best_model.pth'

    # --- 加载数据到内存/显存 ---
    print("Loading data...")
    ply_data_dict = load_ply_dict_to_ram(ply_folder_path, nb_points=150)
    gpu_1d = lode1d_to_gpu(fingerprint_path, device=device)
    gpu_2d = lode1d_to_gpu(drug_2d_path, device=device)

    # --- 构建 Dataset 和 DataLoader ---
    full_dataset = HIVDataset(csv_file_path, gpu_1d, gpu_2d, ply_data_dict)

    # 划分数据集 (8:1:1 或 8:2)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # num_workers=0 因为特征已经在 GPU 上，多进程会报错
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- 初始化模型 ---
    # output_dim=1 表示二分类 (0或1)
    model = MolVisClassifier(nb_views=12, output_dim=1)
    model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # 将参数分为 ACT 部分和其他部分
    # 注意：在你的 MolVisClassifier 中，ACT 嵌套在 ACT_3d_extractor 内
    ACT_params = model.ACT_3d_extractor.ACT.parameters()
    ACT_params_ids = list(map(id, ACT_params))

    # 过滤出非 ACT 的所有其他参数 (包含 1D/2D 分支和融合层)
    other_params = [p for p in model.parameters() if id(p) not in ACT_params_ids]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.lr},  # 主网络学习率
        {'params': model.ACT_3d_extractor.ACT.parameters(), 'lr': args.lr * 0.1}  # ACT 学习率设为 1/10
    ], weight_decay=args.wd)
    # 基于 AUC 的调度器：mode='max' 因为 AUC 越高越好
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=True
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -float('inf')
    best_model_state_dict = None

    # 用于记录结果
    results_list = []

    pbar_epochs = tqdm(total=args.epochs, desc="Training")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        # --- 训练循环 ---
        for batch in train_loader:
            feat_1d = batch['1d'].to(device)
            feat_2d = batch['2d'].to(device)
            points = batch['points'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # 前向
            out = model(feat_1d, feat_2d, points).squeeze()

            # Loss
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ACT_3d_extractor.ACT.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        pbar_epochs.update(1)

        # --- 验证循环 (每 10 epoch) ---
        if epoch % 10 == 0:
            val_loss, auc_val, aupr_val, acc_val, prec_val, rec_val, f1_val = evaluate(val_loader, model, device)
            scheduler.step(auc_val)
            # 打印结果，保持你习惯的格式
            print(f'\nEpoch {epoch} | val_loss:{val_loss:.4f} val_auc:{auc_val:.4f} val_aupr:{aupr_val:.4f} '
                  f'val_accuracy:{acc_val:.4f} val_precision:{prec_val:.4f} val_recall:{rec_val:.4f} val_f1:{f1_val:.4f}')

            # 保存最佳模型
            if auc_val > best_auc:
                best_auc = auc_val
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, model_save_path)
                # torch.save(model, 'model_path/HIV_best_model_full.pth') # 可选

            # 记录数据用于后续 DataFrame
            results_list.append({
                "epoch": epoch,
                "val_auc": auc_val,
                "val_aupr": aupr_val,
                "val_accuracy": acc_val,
                "val_precision": prec_val,
                "val_recall": rec_val,
                "val_f1": f1_val
            })

    # ==========================================
    # 5. 测试 (加载最佳模型进行最终测试)
    # ==========================================
    print("\nTraining Finished. Testing best model...")
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)

    test_loss, auc_test, aupr_test, acc_test, prec_test, rec_test, f1_test = evaluate(test_loader, model, device)

    print(f'Test Results: test_auc:{auc_test:.4f} test_aupr:{aupr_test:.4f} test_accuracy:{acc_test:.4f} '
          f'test_precision:{prec_test:.4f} test_recall:{rec_test:.4f} test_f1:{f1_test:.4f}')

    # 保存最终结果到 csv (可选)
    df_res = pd.DataFrame(results_list)
    df_res.to_csv('training_log.csv', index=False)


if __name__ == '__main__':
    main()
