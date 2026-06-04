# import os
# import sys
# from rdkit import Chem
# import numpy as np
#
#
# def linear_interpolation(points, factor=2):
#     """
#     使用线性插值法在点云之间增加点的密度。
#     factor：每对相邻点之间插入的点的数量（factor > 1 时，表示密度的倍数）。
#     """
#     if len(points) < 2:
#         return points
#
#     new_points_list = [points[0]]
#     for i in range(len(points) - 1):
#         p1 = points[i]
#         p2 = points[i + 1]
#         for j in range(1, factor):
#             interp_point = p1 + (p2 - p1) * (j / float(factor))
#             new_points_list.append(interp_point)
#         new_points_list.append(p2)
#
#     return np.array(new_points_list)
#
#
# def sdf_to_pointcloud_ply(
#         sdf_file: str,
#         ply_file: str,
#         include_hydrogen: bool = True,
#         density_factor: int = 2
# ):
#     """
#     将 SDF 中第一个分子的原子坐标导出为 PLY 点云（ASCII），同时增加点云的密度。
#     - include_hydrogen: 是否包含氢原子
#     - density_factor: 插值因子，决定每对原始点之间插入的点的数量
#     """
#     # 检查输入文件是否存在
#     if not os.path.exists(sdf_file):
#         print(f"警告：输入文件不存在 {sdf_file}", file=sys.stderr)
#         return
#
#     suppl = Chem.SDMolSupplier(sdf_file, removeHs=not include_hydrogen)
#     if not suppl:
#         print(f"警告：无法从 {sdf_file} 读取到任何分子。", file=sys.stderr)
#         return
#
#     mol = suppl[0]
#     if mol is None:
#         print(f"警告：从 {sdf_file} 读取到的第一个条目是无效分子。", file=sys.stderr)
#         return
#
#     if mol.GetNumConformers() == 0:
#         print(f"警告：文件 {sdf_file} 中的分子缺少3D构象。", file=sys.stderr)
#         return
#
#     conf = mol.GetConformer()
#     points = []
#
#     atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
#
#     # 收集原子坐标
#     for idx in atom_indices:
#         atom = mol.GetAtomWithIdx(idx)
#         pos = conf.GetAtomPosition(idx)
#         points.append((pos.x, pos.y, pos.z))
#
#     if not points:
#         print(f"警告：从 {sdf_file} 未能提取任何原子坐标。", file=sys.stderr)
#         return
#
#     points = np.array(points)
#
#     # 如果需要，进行插值
#     if density_factor > 1:
#         # 暂时不插值颜色，因为线性插值颜色意义不大
#         # 如果需要，可以为新点分配相邻点的颜色
#         points = linear_interpolation(points, factor=density_factor)
#
#     n = len(points)
#
#     # 写 PLY 文件
#     with open(ply_file, "w") as f:
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {n}\n")
#         f.write("property float x\n")
#         f.write("property float y\n")
#         f.write("property float z\n")
#         f.write("end_header\n")
#
#         for (x, y, z) in points:
#             f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
#
#
# # --- 主程序入口 ---
# if __name__ == "__main__":
#     # --- 请在这里配置您的输入和输出文件夹 ---
#     input_dir = "./data/bace/drug_sdf"
#     output_dir = "./data/bace/ply_output"
#
#     # --- 配置转换参数 ---
#     INCLUDE_HYDROGEN = True
#     DENSITY_FACTOR = 3
#
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"输入目录: {input_dir}")
#     print(f"输出目录: {output_dir}\n")
#
#     # 遍历输入目录中的所有文件
#     file_count = 0
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".sdf"):
#             file_count += 1
#             input_sdf_file = os.path.join(input_dir, filename)
#
#             # 创建对应的输出文件名
#             output_filename = os.path.splitext(filename)[0] + ".ply"
#             output_ply_file = os.path.join(output_dir, output_filename)
#
#             print(f"正在处理: {filename} -> {output_filename}")
#
#             # 使用 try-except 来捕捉并报告单个文件的错误，避免程序中断
#             try:
#                 sdf_to_pointcloud_ply(
#                     input_sdf_file,
#                     output_ply_file,
#                     include_hydrogen=INCLUDE_HYDROGEN,
#                     density_factor=DENSITY_FACTOR
#                 )
#             except Exception as e:
#                 print(f"处理文件 {filename} 时发生错误: {e}", file=sys.stderr)
#
#     if file_count == 0:
#         print("未在输入目录中找到任何 .sdf 文件。")
#     else:
#         print(f"\n处理完成！共处理了 {file_count} 个文件。")

import os
import sys
from rdkit import Chem
import numpy as np


def sdf_to_pointcloud_ply(
        sdf_file: str,
        ply_file: str,
        include_hydrogen: bool = True
):
    """
    将 SDF 中第一个分子的原子坐标导出为 PLY 点云（ASCII）。
    - include_hydrogen: 是否包含氢原子
    """
    # 检查输入文件是否存在
    if not os.path.exists(sdf_file):
        print(f"警告：输入文件不存在 {sdf_file}", file=sys.stderr)
        return

    suppl = Chem.SDMolSupplier(sdf_file, removeHs=not include_hydrogen)
    if not suppl:
        print(f"警告：无法从 {sdf_file} 读取到任何分子。", file=sys.stderr)
        return

    mol = suppl[0]
    if mol is None:
        print(f"警告：从 {sdf_file} 读取到的第一个条目是无效分子。", file=sys.stderr)
        return

    if mol.GetNumConformers() == 0:
        print(f"警告：文件 {sdf_file} 中的分子缺少3D构象。", file=sys.stderr)
        return

    conf = mol.GetConformer()
    points = []

    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]

    # 收集原子坐标
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        points.append((pos.x, pos.y, pos.z))

    if not points:
        print(f"警告：从 {sdf_file} 未能提取任何原子坐标。", file=sys.stderr)
        return

    points = np.array(points)

    n = len(points)

    # 写 PLY 文件
    with open(ply_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for (x, y, z) in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在这里配置您的输入和输出文件夹 ---
    input_dir = "/data/Classification/General/KPCD3/drug_sdf/"
    output_dir = "/data/Classification/General/KPCD3/ply_output"

    # --- 配置转换参数 ---
    INCLUDE_HYDROGEN = True

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}\n")

    # 遍历输入目录中的所有文件
    file_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".sdf"):
            file_count += 1
            input_sdf_file = os.path.join(input_dir, filename)

            # 创建对应的输出文件名
            output_filename = os.path.splitext(filename)[0] + ".ply"
            output_ply_file = os.path.join(output_dir, output_filename)

            print(f"正在处理: {filename} -> {output_filename}")

            # 使用 try-except 来捕捉并报告单个文件的错误，避免程序中断
            try:
                sdf_to_pointcloud_ply(
                    input_sdf_file,
                    output_ply_file,
                    include_hydrogen=INCLUDE_HYDROGEN
                )
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}", file=sys.stderr)

    if file_count == 0:
        print("未在输入目录中找到任何 .sdf 文件。")
    else:
        print(f"\n处理完成！共处理了 {file_count} 个文件。")


