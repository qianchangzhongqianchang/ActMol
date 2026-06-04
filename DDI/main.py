from get_graph import *
from torch_geometric.loader import NeighborSampler,GraphSAINTRandomWalkSampler
from graph_model import *
import argparse
from tqdm import tqdm
from torch_geometric.data import Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utiles import lode1d_to_gpu, get_drug_2d_features, load_ply_dict_to_ram, get_batch_drug_points
from config import device
import random
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

print(f"Using device: {device}")  
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2000, help="Random seed for model and dataset.")
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default= 0.0069653640155278775, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=5.453697008395925e-05, help='weight decay in optimizer')

args = parser.parse_args()

best_val_loss = float('inf') 




def get_drug_features(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []

    for node_idx in node_idxs:
        drug_name = idx_to_node.get(node_idx.item(), None)

        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:

            drug_feature.append(drug_features)
        else:
            print(f"No features found for {drug_name}. Skipping.")

            continue

    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature


def to_3d(x):
    return rearrange(x, '  c d h w -> (c d) h w ')

def get_drug_features3d(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []

    for node_idx in node_idxs:
        drug_name = idx_to_node.get(node_idx.item(), None)
        drug_name = drug_name+ '_output_encoded.npy'

        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:

            drug_feature.append(drug_features)

        else:
            print(f"No features found for {drug_name}. 3dSkipping.")

            continue


    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature





def test(model, gpu_1d, gpu_2d, gpu_data_3d, best_model_state_dict):
    model.eval()
    model.load_state_dict(best_model_state_dict) 
    test_graph, idx_to_node, node_to_idx = get_graph()

    with torch.no_grad():
        # 获取药物特征
        drug_1d_features_val = get_drug_features(test_graph.node_idx, idx_to_node, gpu_1d)
        drug_2d_features_val = get_drug_features(test_graph.node_idx, idx_to_node, gpu_2d)
        drug_3d_features_val = get_drug_features3d(test_graph.node_idx, idx_to_node, gpu_data_3d)
        out_val = model(test_graph, drug_1d_features_val, drug_2d_features_val, drug_3d_features_val)
    return 0  



def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_predictions_to_csv(scores, labels, csv_file):
    # 将 scores 转换为概率
    probs = torch.sigmoid(scores).flatten()  # 应用 Sigmoid 激活函数，将 logits 转换为概率
    labels = labels.flatten()

    # 创建 DataFrame 保存分数和标签
    df = pd.DataFrame({
        'scores': probs.cpu().numpy(),  # 保存 Sigmoid 后的概率
        'labels': labels.cpu().numpy()  # 保存真实标签
    })

    # 如果文件不存在，创建文件并写入列名（第一次写入时会创建文件并写入列名）
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False, mode='w', header=True)
    else:
        # 否则，追加到已有文件中，不写入列名
        df.to_csv(csv_file, index=False, mode='a', header=False)



def save_best_model(model, best_model_state_dict, epoch, save_dir='./models'):
    """ 保存最佳模型的权重 """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, f'best_model_epoch_{epoch}.pth')
    torch.save(best_model_state_dict, model_path)
    print(f"Best model saved to {model_path}.")


def evaluate_for_plot(val_data, model, device, idx_to_node, gpu_1d, gpu_2d, ply_data_dict):
    model.eval()
    with torch.no_grad():
        # 1. 准备特征数据 (保持与训练时一致)
        drug_1d_features_val = get_drug_features(val_data.node_idx, idx_to_node, gpu_1d)
        drug_2d_features_val = get_drug_features(val_data.node_idx, idx_to_node, gpu_2d)
        val_points = get_batch_drug_points(val_data.node_idx.tolist(), ply_data_dict, device, idx_to_node)

        # 2. 模型前向传播
        out_val, _ = model(val_data, drug_1d_features_val, drug_2d_features_val, val_points)

        # 3. 调用模型内部的 compute_loss 获取 logits (val_scores) 和 labels
        # 注意：这里要确保返回的是没有经过 sigmoid 的原始得分或处理后的 scores
        _, val_scores, val_labels, _, _ = model.compute_loss(out_val, val_data)

        # 4. 处理为绘图所需的格式
        # 使用 sigmoid 将 logits 转换为 0-1 之间的概率
        # 使用 .ravel() 将其展平为一维数组，防止 sklearn 报错
        val_probs = torch.sigmoid(val_scores).cpu().numpy().ravel()
        val_true = val_labels.cpu().numpy().ravel()

    return val_true, val_probs

def plot_curves(y_true, y_probs, save_path_prefix='test_metrics'):
    """绘制 ROC 和 PR 曲线并保存"""
    plt.figure(figsize=(12, 5))

    # --- ROC Curve ---
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # --- PR Curve ---
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}.png', dpi=300)
    print(f"Curves saved to {save_path_prefix}.png")
    plt.close()

def main():
    seed = args.seed
    set_random_seed(seed)
    ply_folder_path = '/data/Classification/DDI/ply_output/'
    ply_data_dict = load_ply_dict_to_ram(ply_folder_path, nb_points=100)
    # npy_files_dir = 'ddi_encoded'
    # npy_files = [os.path.join(npy_files_dir, file) for file in os.listdir(npy_files_dir) if file.endswith('.npy')]
    # gpu_data_3d = load_npy_to_gpu(npy_files, device)
    gpu_1d = lode1d_to_gpu('/data/Classification/DDI/drug_1d_fingerprints.csv',device=device)
    gpu_2d = lode2d_to_gpu('/data/Classification/DDI/drug_2d.csv',device=device)
    train_data, val_data, test_data,idx_to_node= get_graph()

    model = ActMol()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    
    model.to(device)
    best_model_state_dict = None
    best_val_loss = float('inf')
    best_auc = -float('inf')
    pbar_epochs = tqdm(total=args.epochs, desc=f"Training", leave=False)
    #csv_file = './predictions/predictions5.csv'
    for epoch in range(args.epochs):
        model.train()  
        drug_1d_features = get_drug_features(train_data.node_idx,idx_to_node,gpu_1d)
        drug_2d_features = get_drug_features(train_data.node_idx,idx_to_node,gpu_2d)
        # drug_3d_features = get_drug_features3d(train_data.node_idx,idx_to_node,gpu_data_3d)
        drug_points = get_batch_drug_points(train_data.node_idx.tolist(), ply_data_dict, device,idx_to_node)
        # out,weight= model(train_data,drug_1d_features,drug_2d_features,drug_3d_features)
        out, weight = model(train_data, drug_1d_features, drug_2d_features, drug_points)
        # print(drug_1d_features.shape)
        # print(drug_2d_features.shape)
        loss,scores, labels, edge,_= model.compute_loss(out, train_data)
        loss = loss 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        pbar_epochs.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        pbar_epochs.update(1)
        if (epoch ) % 10 == 0:
            # model.eval()
            with torch.no_grad():
                drug_1d_features_val = get_drug_features(val_data.node_idx,idx_to_node,gpu_1d)
                drug_2d_features_val = get_drug_features(val_data.node_idx,idx_to_node,gpu_2d)
                # drug_3d_features_val = get_drug_features3d(val_data.node_idx,idx_to_node,gpu_data_3d)
                val_points = get_batch_drug_points(val_data.node_idx.tolist(), ply_data_dict, device,idx_to_node)
                out_val,weight_val = model(val_data,drug_1d_features_val,drug_2d_features_val,val_points)
                val_loss,val_scores,  val_labels, edge,t = model.compute_loss(out_val, val_data)
                auc_val,aupr_val,accuracy_val,precision_val,recall = model.test(val_scores, val_labels)
                print('val_loss',val_loss.item())
                print('auc_val',auc_val)
                print('accuracy_val',accuracy_val)
                print('precision_val',precision_val)

                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_model_state_dict = model.state_dict()
                    save_best_model(model, best_model_state_dict, epoch)
                    #save_predictions_to_csv(val_scores, val_labels, csv_file)
                    # 1. 运行验证集推断获取概率和标签
                    # val_true, val_probs = evaluate_for_plot(
                    #     val_data,
                    #     model,
                    #     device,
                    #     idx_to_node,
                    #     gpu_1d,
                    #     gpu_2d,
                    #     ply_data_dict
                    # )
                    val_probs_for_plot = torch.sigmoid(val_scores).cpu().numpy().ravel()
                    val_true_for_plot = val_labels.cpu().numpy().ravel()

                    print(f"Plotting AUC using same data: {auc_val:.4f}")


                    # fpr, tpr, _ = roc_curve(val_true, val_probs)
                    #
                    # current_auc = auc(fpr, tpr)
                    # print(f"Plotting AUC: {current_auc:.4f}")
                    # # 3. 保存该最佳 Epoch 的曲线数据到 CSV
                    # pred_df = pd.DataFrame({
                    #     'y_true': val_true_for_plot,
                    #     'y_prob': val_probs_for_plot
                    # })
                    # # # 文件名建议带上 epoch 号，方便溯源
                    # pred_df.to_csv(f'roc_data_val_best_epoch_{epoch}.csv', index=False)
                #     fpr, tpr, _ = roc_curve(val_true_for_plot, val_probs_for_plot)
                #     roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                #     roc_df.to_csv(f'roc_data_best_epoch_{epoch}.csv', index=False)
                #     # 4. 调用绘图函数（保存到特定目录）
                #     print(f"New Best AUC found at Epoch {epoch}! Saving plots...")
                #     plot_curves(val_true_for_plot, val_probs_for_plot, save_path_prefix='plots_val_best')  # 1. 运行验证集推断获取概率和标签
                # results = {
                #
                #     "avg_val_auc": [auc_val],
                #     "avg_val_aupr": [aupr_val],
                #     "avg_val_accuracy": [accuracy_val],
                #     "avg_val_precision": [precision_val],
                #     "avg_val_recall": [recall],
                # }
                # print(results)

main()
