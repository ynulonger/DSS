import cv2
import math
import time
import torch
import anndata
import scanpy as sc
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.ndimage import generic_filter
from sklearn.preprocessing import normalize
from scipy.ndimage import label, find_objects
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def draw_star_point(img, centers, labels):
    print(f"num of points sampled: {len(labels)}, {img.shape}")
    outer_radius = 10  # 外接圆半径
    inner_radius = 3   # 内接圆半径
    n_points = 5        # 五角星
    # 计算顶点坐标
    for center, label in zip(centers, labels):
        points = []
        for i in range(2 * n_points):
            angle = i * math.pi / n_points  # 角度（弧度）
            radius = inner_radius if i % 2 == 1 else outer_radius  # 交替内外半径
            x = center[0] + radius * math.cos(angle - math.pi/2)  # 减去 π/2 让星形正立
            y = center[1] + radius * math.sin(angle - math.pi/2)
            points.append((int(x), int(y)))

    # 转换为NumPy数组
        pts = np.array(points, dtype=np.int32)
        # 绘制星形（填充）
        cv2.fillPoly(img, [pts], color=(255, 0, 0) if label==1 else (0, 255, 0))
    return img

def mode_filter(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]

def min_max_normalize(data):
    """ 将每列线性缩放到 [0,1] 范围 """
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return (data - mins) / (maxs - mins + 1e-8)  # 加小量避免除零

# def binary_spatial_entropy_map(binary_image, window_size=3):
#     """
#     计算二值图像的局部空间熵图
#     binary_image: 2D numpy array, dtype=bool 或 {0,1}
#     window_size: 滑动窗口大小（建议奇数，如 3, 5）
#     """
#     # 确保是 0/1 或 bool
#     img = binary_image.astype(bool)
#     # 定义局部熵函数
#     def local_entropy(window):
#         # 统计 0 和 1 的数量
#         count_0 = np.sum(window == 0)
#         count_1 = np.sum(window == 1)
#         total = count_0 + count_1
        
#         if total == 0:
#             return 0.0
        
#         p0 = count_0 / total
#         p1 = count_1 / total
        
#         entropy = 0.0
#         if p0 > 0:
#             entropy -= p0 * np.log2(p0)
#         if p1 > 0:
#             entropy -= p1 * np.log2(p1)
        
#         return entropy  # 最大为 1.0（当 p0=p1=0.5）

#     # 使用通用滤波器滑动窗口
#     pad = window_size // 2
#     padded = np.pad(img, pad, mode='constant', constant_values=0)
#     entropy_map = np.zeros_like(img, dtype=np.float32)
    
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             window = padded[i:i+window_size, j:j+window_size]
#             entropy_map[i, j] = local_entropy(window)
#     return entropy_map

def binary_spatial_entropy_map(binary_image, window_size=3):
    """
    使用OpenCV进行更快的计算二值图像的局部空间熵图
    binary_image: 2D numpy array, dtype=bool 或 {0,1}
    window_size: 滑动窗口大小（建议奇数，如 3, 5）
    """
    img = binary_image.astype(np.uint8)
    
    # 使用OpenCV的boxFilter进行快速卷积
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    count_1 = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size), 
                           borderType=cv2.BORDER_CONSTANT)
    count_1 *= window_size * window_size  # 因为boxFilter是平均滤波，需要乘以总数
    
    total_pixels = window_size * window_size
    count_0 = total_pixels - count_1
    # 计算概率和熵
    p0 = count_0 / total_pixels
    p1 = count_1 / total_pixels
    entropy = np.zeros_like(p0, dtype=np.float32)
    # 使用向量化计算
    mask = (p0 > 0) & (p1 > 0)
    entropy[mask] = -p0[mask] * np.log2(p0[mask]) - p1[mask] * np.log2(p1[mask])
    return entropy


def spatial_smoothness_8n(labels):
    """
    计算空间平滑性指标（8邻域版本）。
    参数：
        labels: 1D array，长度为 height * width，对应每个 patch 的类别标签
        height: 图像高度（patch 格数）
        width: 图像宽度（patch 格数）
    返回：
        平滑性得分（越高表示标签在空间上越连续）
    """
    height, width = labels.shape
    total = 0
    same = 0

    # 8邻域坐标偏移
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    for i in range(height):
        for j in range(width):
            for dx, dy in neighbors:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:
                    total += 1
                    if labels[i, j] == labels[ni, nj]:
                        same += 1

    return same / total if total > 0 else 0

def get_candidate_fg_clusters(image):
    unique_values = np.unique(image)
    structure_8 = np.ones((3, 3), dtype=int)
    candidate_clusters = []
    # print(f'FG list: {unique_values}')
    for target_value in unique_values:
        # 举例：只对像素值为 0 的区域进行分析
        mask = (image == target_value).astype(int)
        labeled_array, num_component = label(mask, structure=structure_8)
        if num_component<50:
            candidate_clusters.append(target_value)
    return candidate_clusters

from scipy.ndimage import label
from skimage.morphology import remove_small_objects

def remove_small_regions(mask, min_size=20):
    """
    mask: 2D binary np.ndarray
    return: mask with small connected components removed
    """
    cleaned_mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return cleaned_mask.astype(int)

def get_sim_map(adata, label, fg_cluster, H, W, pad_H, pad_W):
    X_PCA = torch.from_numpy(adata.obsm['X_pca'].copy())
    if fg_cluster==None:
        idx_bool = (label.ravel() != fg_cluster)
        X_PCA_filtered = X_PCA
    else:
        idx_bool = (label.ravel() == fg_cluster)
        X_PCA_filtered = X_PCA[idx_bool]
    regional_mean_feat = X_PCA_filtered.mean(dim=0, keepdim=True)
    similarity = F.cosine_similarity(regional_mean_feat, X_PCA).reshape(H,W).numpy()
    sim_map, _, _ = smooth_and_unpad(similarity, pad_H, pad_W, smooth=False)
    sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    return sim_map_norm

def get_leiden_labels_sim_maps(
    data,
    pad_H, pad_W,
    resolution=0.5,
    n_neighbors=None,
    n_pca=64
):
    C, H, W = data.shape
    X = data.reshape(C, -1).T  # (N, 768)
    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    N = H * W
    sim_maps = []

    if n_neighbors is None:
        n_neighbors = int((H * W) ** 0.5)
    # Step 1: 初始 Leiden 聚类（提供多类先验结构）
    adata = anndata.AnnData(X.copy())
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pca)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=n_pca, method='gauss')
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_init')
    initial_labels = adata.obs['leiden_init'].astype(int).values
    labels_map = initial_labels.reshape(H, W)
    smoothed_label_image, init_smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map, pad_H, pad_W)
    candidate_fgs = get_candidate_fg_clusters(smoothed_label_image)
    for fg_cluster in candidate_fgs:
        sim_map_leiden = get_sim_map(adata, init_smoothed_label_image_low_res, fg_cluster, H, W, pad_H, pad_W)
        # sim_map_leiden = cv2.GaussianBlur(sim_map_leiden, (5, 5), sigmaX=0)
        sim_maps.append(sim_map_leiden)
    return sim_maps

def get_patch_level_hierarchical_labels(
    data,
    pad_H, pad_W,
    resolution=0.5,
    n_neighbors=None,
    n_pca=64,
    alpha=1.0,   # 特征距离权重
    beta=1.5,     # 空间平滑权重
    gamma = 0.5
):
    """
    ### Adaptive Stopping Criterion），使得：
        ✅ 不浪费计算资源（不迭代过度）
        ✅ 不提前终止（保证收敛）
        ✅ 适用于不同图像（大小、复杂度不同)
    基于层次化思想的 patch-level 迭代优化
    - 不合并整个 cluster，而是逐 patch 更新
    - 每轮利用前一轮的“类结构”作为先验

    """
    C, H, W = data.shape
    X = data.reshape(C, -1).T  # (N, 768)
    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    N = H * W
    leiden_iter_labels = []
    leiden_iter_labels_low_res = []
    smoothed_padded_masks = []
    sim_maps_leiden_list = []
    sim_maps_refine_list = []

    measurements = []
    energies = []
    if n_neighbors is None:
        n_neighbors = int((H * W) ** 0.5)

    # Step 1: 初始 Leiden 聚类（提供多类先验结构）
    adata = anndata.AnnData(X.copy())
    start = time.time()
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pca)
    # '''
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=n_pca, method='gauss')

    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_init', flavor="igraph")

    # sc.tl.umap(adata)
    initial_labels = adata.obs['leiden_init'].astype(int).values
    labels_map = initial_labels.reshape(H, W)
    leiden_map, init_smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map, pad_H, pad_W)
    candidate_fgs = get_candidate_fg_clusters(leiden_map)
    candidate_fgs_updated = []
    # filter out outlier
    threshold = int(init_smoothed_label_image_low_res.shape[0]*init_smoothed_label_image_low_res.shape[1]*0.01)

    leiden_iter_labels_low_res.append(init_smoothed_label_image_low_res)
    smoothed_padded_masks.append(smoothed_padded)
    # Step 3: 构建空间-特征图（用于邻居传播）
    dist_spatial = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    spatial_weight = np.exp(-dist_spatial ** 2 / (2 * (beta)**2))
    feat_sim = cosine_similarity(X)
    sim_joint = feat_sim * spatial_weight
    np.fill_diagonal(sim_joint, -1)
    knn_indices = np.argsort(sim_joint, axis=1)[:, -int(np.sqrt(N)):]
    # Step 4: 迭代优化（每轮更新 patch 标签）
    # Step 2: 初始化二值标签 
    # print('candidate foregrounds:', candidate_fgs)
    # sim_maps_img = get_sim_map(adata, init_smoothed_label_image_low_res, None, H, W, pad_H, pad_W)
    for fg_cluster in candidate_fgs:
        # print('optimizing fg', fg_cluster)
        y = (init_smoothed_label_image_low_res.reshape(-1) == fg_cluster).astype(float)  # 软标签 (0~1)
        delta = 10 
        it = 1
        while delta > 1:
        # for it in range(n_iterations):
            # 1. 更新前景/背景中心（只使用高置信度 patch）
            fg_mask = (y > 0.5)
            bg_mask = (y < 0.5)
            
            if np.any(fg_mask) and np.any(bg_mask):
                center_fg = np.average(adata.obsm['X_pca'][fg_mask], axis=0, weights=y[fg_mask])
                center_bg = np.average(adata.obsm['X_pca'][bg_mask], axis=0, weights=1 - y[bg_mask])
            else:
                center_fg = adata.obsm['X_pca'][fg_mask].mean(axis=0) if np.any(fg_mask) else adata.obsm['X_pca'].mean(axis=0)
                center_bg = adata.obsm['X_pca'][bg_mask].mean(axis=0) if np.any(bg_mask) else adata.obsm['X_pca'].mean(axis=0)

            # 2. 计算每个 patch 的得分
            y_new = np.zeros(N)

            # ------- 1. 特征得分 -------
            # 计算每个点到前景/背景中心的距离
            d_fg = np.linalg.norm(adata.obsm['X_pca'] - center_fg, axis=1)  # shape (N,)
            d_bg = np.linalg.norm(adata.obsm['X_pca'] - center_bg, axis=1)  # shape (N,)

            score_feat = d_bg - d_fg   # shape (N,)

            # ------- 2. 邻域一致性得分 -------
            # neighbor_labels: (N, K)，K 是邻居个数
            neighbor_labels = y[knn_indices]   # 直接广播取值
            score_smooth = neighbor_labels.mean(axis=1)  # shape (N,)

            # ------- 3. 综合得分 -------
            total_score = alpha * score_feat + beta * score_smooth

            # ------- 4. sigmoid 转换 -------
            y_new = 1.0 / (1.0 + np.exp(-total_score))
            y = y_new


            energy = compute_energy(y, X, knn_indices)
            energies.append(energy)
            it += 1
            if len(energies) >=2:
                # 最近两次能量变化很小
                delta = np.abs(energies[-1] - energies[-2])
                if delta < 1:
                    # Step 5: 二值化 + 平滑
                    binary_labels = (y > 0.5).astype(int)
                    labels_map = binary_labels.reshape(H, W)
                    labels_map = remove_small_regions(labels_map, min_size=threshold)

                    smoothed_label_image, smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map, pad_H, pad_W)
                    
                    if len(np.unique(smoothed_label_image_low_res)) ==1:
                        # sim_map_leiden = get_sim_map(adata, init_smoothed_label_image_low_res, fg_cluster, H, W, pad_H, pad_W)
                        # sim_map_leiden = cv2.GaussianBlur(sim_map_leiden, (5, 5), sigmaX=0)
                        # sim_maps_leiden_list.append(sim_map_leiden)
                        continue
                    # else:
                    sim_map_leiden = get_sim_map(adata, init_smoothed_label_image_low_res, fg_cluster, H, W, pad_H, pad_W)
                    sim_map_leiden = cv2.GaussianBlur(sim_map_leiden, (5, 5), sigmaX=0)
                    sim_maps_leiden_list.append(sim_map_leiden)
                    
                    candidate_fgs_updated.append(fg_cluster)
                    leiden_iter_labels.append(smoothed_label_image)
                    leiden_iter_labels_low_res.append(smoothed_label_image_low_res)
                    smoothed_padded_masks.append(smoothed_padded)

                    sim_map_opt = get_sim_map(adata, smoothed_label_image_low_res, 1, H, W, pad_H, pad_W)
                    sim_map_opt = cv2.GaussianBlur(sim_map_opt, (5, 5), sigmaX=0)
                    sim_maps_refine_list.append(sim_map_opt)
                    # break
    

    pca_data = np.reshape(adata.obsm['X_pca'], [H,W, n_pca])
    return leiden_iter_labels, candidate_fgs_updated, sim_maps_leiden_list, sim_maps_refine_list, smoothed_padded_masks, leiden_iter_labels_low_res, leiden_map, pca_data


def compute_fisher_score(X, labels):
    X1 = X[labels == 0]
    X2 = X[labels == 1]
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    var1 = X1.var(axis=0)
    var2 = X2.var(axis=0)
    numerator = np.square(mean1 - mean2)           # (D,)
    denominator = var1 + var2 + 1e-8               # 避免除0
    fisher_scores = numerator / denominator        # (D,)
    average_fisher_score = np.mean(fisher_scores)  # 可选：对所有维度求平均
    return average_fisher_score

def compute_energy(y, X, knn_indices, alpha=1.0, beta=1.):
    """
    能量越低越稳定
    - 类内紧凑
    - 类间分离
    - 空间平滑
    """
    fg_mask = (y > 0.5)
    bg_mask = (y < 0.5)
    energy = 0.0
    
    # 1. 类间分离（负的类间距离）
    if np.any(fg_mask) and np.any(bg_mask):
        center_fg = X[fg_mask].mean(axis=0)
        center_bg = X[bg_mask].mean(axis=0)
        inter_class_sep = alpha * np.linalg.norm(center_fg - center_bg)
        energy -= inter_class_sep
    
    # 2. 空间平滑（邻居差异小）
    inner_class = 0.0
    for i in range(len(y)):
        neighbor_diff = np.mean(np.abs(y[i] - y[knn_indices[i]]))
        inner_class += neighbor_diff
    energy += beta * inner_class / len(y)
    # print(f'inter_class:{inter_class_sep}, inner_class:{inner_class}')
    return energy

def calculate_candidate_mask_score(data, sort_directions):
    """
    计算每个特征上样本的排序排名，允许为每个特征指定不同排序方向
    
    参数:
    data (np.ndarray): 样本特征矩阵，形状为 (n_samples, n_features)
    sort_directions (list or np.ndarray): 每个特征的排序方向，长度等于n_features
        True表示升序（从小到大），False表示降序（从大到小）
    
    返回:
    np.ndarray: 排名矩阵，形状与输入相同，包含每个样本在每个特征上的排名
    """
    data = min_max_normalize(data)
    n_samples, n_features = data.shape
    scores = np.zeros_like(data, dtype=float)
    # 检查方向参数是否合法
    if len(sort_directions) != n_features:
        raise ValueError("sort_directions的长度必须与特征数量一致")
    # 遍历每个特征
    for feature_idx in range(n_features):
        # 获取当前特征的所有值和排序方向
        feature_values = data[:, feature_idx]
        ascending = sort_directions[feature_idx]
        if ascending:
            scores[:,feature_idx] = 1-feature_values
        else:
            scores[:,feature_idx] = feature_values    
    scores = scores.sum(axis=1)
    return scores

def smooth_and_unpad(labels_map, pad_H, pad_W, size=5, smooth=True):
    if smooth:
        smoothed_label_image_low_res = generic_filter(labels_map, function=mode_filter, size=size)
    else:
        smoothed_label_image_low_res = labels_map
    smoothed_label_image = cv2.resize(smoothed_label_image_low_res, dsize=None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    # 去除 padding
    if pad_H == 0:
        if pad_W == 0:
            smoothed_unpad_label = smoothed_label_image
            pass
        else:
            smoothed_unpad_label = smoothed_label_image[:, :-pad_W]
    else:
        if pad_W == 0:
            smoothed_unpad_label = smoothed_label_image[:-pad_H, :]
        else:
            smoothed_unpad_label = smoothed_label_image[:-pad_H, :-pad_W]
    return smoothed_unpad_label, smoothed_label_image_low_res, smoothed_label_image
      

def sample_heatmap_points(heatmap, step=24, high_thresh=0.8, low_thresh=0.2):
    """
    从 heatmap 上稀疏取点并打标签

    参数：
        heatmap (ndarray): n*m 的 2D array，值范围 [0, 1]
        step (int): 网格采样间隔
        high_thresh (float): 高阈值，值 > high_thresh 标记为 1
        low_thresh (float): 低阈值，值 < low_thresh 标记为 0

    返回：
        coords (ndarray): shape = (k, 2)，每一行是 (y, x) 坐标
        labels (list): 长度 = k，每个坐标对应的标签（0 或 1）
    """
    n, m = heatmap.shape
    coords = []
    labels = []

    # 按网格步长采样
    for y in range(0, m, step):
        for x in range(0, n, step):
            val = heatmap[x, y]
            if val > high_thresh:
                coords.append((x, y))
                labels.append(1)
            elif val < low_thresh:
                coords.append((x, y))
                labels.append(0)
            # 中间值 [low_thresh, high_thresh] 直接丢弃

    return np.array(coords), labels
