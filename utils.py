import re
import os
import io
import cv2
import math
import json
import torch
import base64
import random
import anndata
import matplotlib
import numpy as np
import scanpy as sc
from PIL import Image
import torch.nn as nn
from pprint import pprint
# import albumentations as A
import torch.nn.functional as F
from skimage import morphology
from scipy.stats import pearsonr
from refine_leiden_utils import *
import torchvision.transforms as T
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from torchvision.transforms import v2
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
import torchvision.transforms as transforms
from skimage.measure import shannon_entropy
from matplotlib.colors import ListedColormap
from qwen_vl_utils import process_vision_info
from skimage.morphology import dilation, disk
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import local_binary_pattern
from scipy.sparse.csgraph import connected_components
from transformers import AutoModel,AutoImageProcessor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label, find_objects, generate_binary_structure

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_rotated_box(mask):
    # 找连通区域轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 可视化
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rotated_boxes = []
    for cnt in contours:
        if len(cnt) < 5:
            continue  # minAreaRect 需要至少5个点
        
        # 最小外接矩形
        rect = cv2.minAreaRect(cnt)  # (center, (w,h), angle)
        box = cv2.boxPoints(rect)    # 4个顶点
        box = box.astype(int)              # 转换为整数
        box = box.reshape((-1,2))      # reshape 成 (4,1,2)
        # print(box)
        rotated_boxes.append(box)
    return rotated_boxes

def boundary_contact(mask, n=10):
    """
    计算mask与图像边缘的接触比例
    参数:
    mask: 二维numpy数组，表示分割掩码
    n: 整数，表示要考虑的边缘像素宽度（默认为10）
    返回:
    float: 边缘接触比例 (0-1之间)
    """
    H, W = mask.shape
    # 确保n不超过图像尺寸的一半
    n = min(n, H // 2, W // 2)
    # 提取四个边缘区域的n个像素
    top_edge = mask[0:n, :]           # 上边缘n行
    bottom_edge = mask[H-n:H, :]      # 下边缘n行
    left_edge = mask[n:H-n, 0:n]      # 左边缘n列（排除上下角重复部分）
    right_edge = mask[n:H-n, W-n:W]   # 右边缘n列（排除上下角重复部分）
    # 计算每个边缘区域中非零像素的数量
    top_contact = np.sum(top_edge > 0)
    bottom_contact = np.sum(bottom_edge > 0)
    left_contact = np.sum(left_edge > 0)
    right_contact = np.sum(right_edge > 0)
    # 计算总接触像素数
    contact_pixels = top_contact + bottom_contact + left_contact + right_contact
    # 计算边缘区域的总像素数
    total_edge_pixels = (n * W) + (n * W) + (n * (H - 2 * n)) + (n * (H - 2 * n))
    return contact_pixels / total_edge_pixels

def mode_filter(x):
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]

def ResizeLongestSide(image, target_size=980):
    width, height = image.size
    original_longest = max(height, width)
    # 如果最长边已经大于等于目标尺寸，直接返回原图
    # if original_longest >= target_size:
    #     return image
    # 计算缩放比例
    scale = target_size / original_longest
    # 计算新的尺寸
    if height > width:
        new_height = target_size
        new_width = int(width * scale)
    else:
        new_width = target_size
        new_height = int(height * scale)
    # 使用双线性插值进行缩放
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    # print(f'original size: {width, height}, resized: {resized_image.size}')
    return resized_image

# 示例二维灰度图像（像素值在0~6之间）

def get_leiden_label(data, leiden_key, pad_H, pad_W, resolution=0.5):
    # reshape 成 [n_samples, n_features]
    n_features, h, w = data.shape
    C, H, W = data.shape
    # 生成空间坐标（用于空间优化）
    n_neighbors = int((H*W)**0.5)
    n_pca = 64
    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    X = data.reshape(n_features, -1).T  # [3640, 768]
    # Step 2: 构建 AnnData 对象
    print(f'resolution:{resolution}, n_neighbors:{n_neighbors}, n_pca:{n_pca}')
    adata = anndata.AnnData(X)
    # Step 3: 预处理（标准化） + PCA 降维（可选但推荐）
    sc.pp.scale(adata)  # 标准化每个特征
    # 使用 PCA 降维（可选，加速聚类）
    sc.tl.pca(adata, n_comps=64)
    # 构建邻接图
    adata.obsm['spatial'] = coords
    sc.pp.neighbors(adata, 
        use_rep='X_pca', 
        n_neighbors = n_neighbors, 
        n_pcs = n_pca,
        method='gauss',          # 适用于空间数据
        metric='euclidean'    # ✅ 真正利用空间坐标
    )

    # Step 4: Leiden 聚类
    sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key)
    labels = adata.obs[leiden_key].astype(int).values
    labels_map = labels.reshape(h, w)
    smoothed_label_image = generic_filter(labels_map, function=mode_filter, size=5)
    smoothed_label_image = cv2.resize(smoothed_label_image, dsize=None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    if pad_H == 0:
        if pad_W == 0:
            return smoothed_label_image, adata
        else:
            return smoothed_label_image[:, :-pad_W], adata
    else:
        if pad_W == 0:
            return smoothed_label_image[:-pad_H, :], adata
        else:
            return smoothed_label_image[:-pad_H, :-pad_W], adata


def show_image(fig, ax, img, title="", show_bar=True):
    img = np.squeeze(img)
    if show_bar:
        n_classes = len(np.unique(img))
        colors = matplotlib.colormaps["Set3"]
        cmap = ListedColormap(colors(np.linspace(0, 1, n_classes)))
        img_plot = ax.imshow(img, cmap=cmap, vmin=0, vmax=n_classes-1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # size: colorbar 宽度，pad: 间距
        cbar = fig.colorbar(img_plot, cax=cax)
        cbar.set_ticks(range(n_classes))
        cbar.set_label("Clusters")
    else:
        n_classes = len(np.unique(img))
        colors = matplotlib.colormaps["Set3"]
        cmap = ListedColormap(colors(np.linspace(0, 1, n_classes)))
        img_plot = ax.imshow(img, cmap=cmap, vmin=0, vmax=n_classes-1)

    ax.set_title(title)
    ax.axis("off")

def load_and_pad_image(image_path, patch_size=14, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], target_size=1120):
    image = Image.open(image_path).convert("RGB")
    # image = Image.open(image_path)
    Orig_W, Orig_H = image.size
    image = ResizeLongestSide(image, target_size=target_size)
    # 计算需要 padding 的大小
    W, H = image.size
    pad_W = (patch_size - W % patch_size) % patch_size
    pad_H = (patch_size - H % patch_size) % patch_size
    # 定义 padding transform（右边和下边填充）
    pad_transform = transforms.Pad((0, 0, pad_W, pad_H), fill=0)  # 或 fill=(123, 117, 104) 等灰度值
    normalize = transforms.Normalize(mean=mean, std=std)
    # 组合 transform
    transform = transforms.Compose([
        transforms.ToTensor(),         # 转为 [0, 1]
        normalize,                     # 归一化
        pad_transform                  # padding 到 patch size 的整数倍
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, pH, pW]
    return image, image_tensor, (pad_W, pad_H), (Orig_W, Orig_H)
    
def evaluate(model, test_loader, device, criterion_ce, criterion_iou, num_classes=2, ignore_index=2):
    model.eval()
    preds_all = []
    labels_all = []
    loss_ce = 0
    loss_iou = 0
    loss_total = 0

    with torch.no_grad():
        for images, labels, pad_info, names in test_loader:
            images = images.to(device)
            labels = labels.to(device)      # [n, 73, 73], 值为 0,1,2（2=ignore）

            logits = model(images)  # [B, num_classes, 73, 73]

            loss_ce += criterion_ce(logits, labels).item()
            loss_iou += criterion_iou(logits, labels).item()
            loss_total += loss_ce+loss_iou

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # [B, 73, 73]
            preds_all.append(preds.cpu())
            labels_all.append(labels.to('cpu'))

    # 合并所有 batch
    preds_all = torch.cat(preds_all, dim=0)  # (N, H, W)
    labels_all = torch.cat(labels_all, dim=0)  # (N, H, W)
    # 创建 mask，忽略 ignore_index
    valid_mask = (labels_all != ignore_index)  # bool mask
    # 只保留有效位置
    preds_valid = preds_all[valid_mask]
    labels_valid = labels_all[valid_mask]
    # 计算 Accuracy
    acc = (preds_valid == labels_valid).float().mean().item()
    # ---------------------------------------------
    # 🔢 计算 mIoU (mean Intersection over Union)
    # ---------------------------------------------
    iou_score = 0.0
    iou_list = []
    for cls in range(num_classes):
        pred_class = (preds_valid == cls)
        label_class = (labels_valid == cls)
        # 避免除零
        intersection = (pred_class & label_class).sum().float()
        union = (pred_class | label_class).sum().float()
        if union == 0:
            # 该类在 label 和 pred 中都不存在，跳过（或设为 1？）
            iou = float('nan')
        else:
            iou = (intersection + 1e-6) / (union + 1e-6)  # 加小数防止除零
        iou_list.append(iou)

    # 只计算存在的类的平均 IoU（忽略 nan）
    iou_tensor = torch.tensor(iou_list)
    iou_score = iou_tensor.nanmean().item()  # mean 忽略 nan

    # print(f"Test Accuracy: {acc:.4f}")
    # print(f"Test mIoU: {iou_score:.4f}")
    # for cls in range(num_classes):
    #     iou_val = iou_list[cls] if not isinstance(iou_list[cls], float) or not (iou_list[cls] != iou_list[cls]) else float('nan')
    #     print(f"  Class {cls} IoU: {iou_val:.4f}")

    return preds_all, acc, iou_list[1], loss_ce/len(test_loader), loss_iou/len(test_loader), loss_total/len(test_loader)

def mask_to_hard_labels_vectorized(mask_np, patch_size=14, ignore_value=2, thresh_pos=0.3):
    """
    mask_np: (H, W), dtype=int, 值为 0, 1, 2（2=ignore）
    返回: (Ph, Pw) 的 hard label
    """
    H, W = mask_np.shape
    Ph, Pw = H // patch_size, W // patch_size

    # reshape 成 patches: (Ph, Pw, 14, 14)
    patches = mask_np.reshape(Ph, patch_size, Pw, patch_size)
    patches = patches.transpose(0, 2, 1, 3)  # -> (Ph, Pw, 14, 14)
    patches = patches.reshape(Ph, Pw, -1)    # (Ph, Pw, 196)

    # 有效像素 mask（非 ignore）
    valid = (patches != ignore_value)  # (Ph, Pw, 196)
    has_valid = valid.sum(axis=-1) > 0   # (Ph, Pw)

    # 提取有效区域的值（前景=1）
    values = patches.copy()
    values[~valid] = 0  # 无效位置置 0
    ratio = (values == 1).sum(axis=-1).astype(np.float32) / (valid.sum(axis=-1) + 1e-8)

    # 初始化为 ignore
    labels = np.full((Ph, Pw), fill_value=ignore_value, dtype=np.int32)

    # 设置前景和背景
    labels[has_valid & (ratio >= thresh_pos)] = 1
    labels[has_valid & (ratio <= thresh_pos)] = 0
    upsampled_labels = cv2.resize(labels.astype(np.uint8), (0,0), fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    return upsampled_labels

def resize_and_pad_1(image: Image.Image, target_size=1022, is_mask=True):
    """
    同时对图像和 mask 进行等比缩放 + pad
    image: PIL Image (RGB)
    mask: PIL Image (L), 灰度图（0-255）
    返回: image_pil, mask_pil (都为 PIL Image，已 pad，mask 的 pad 区域为 ignore_value)
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 缩放
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    ignore_value = 2 if is_mask else 0
    image = image.resize((new_w, new_h), resample=resample)
    # 计算 padding
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    right = target_size - new_w - left
    bottom = target_size - new_h - top
    pad_info = np.array([left, top, right, bottom])
    image = T.Pad((left, top, right, bottom), fill=ignore_value)(image)
    return image, pad_info

def resize_and_pad(image: Image.Image, mask: Image.Image, target_size=1022, ignore_value=2):
    """
    同时对图像和 mask 进行等比缩放 + pad
    image: PIL Image (RGB)
    mask: PIL Image (L), 灰度图（0-255）
    返回: image_pil, mask_pil (都为 PIL Image，已 pad，mask 的 pad 区域为 ignore_value)
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 缩放
    # print('before resize:', image.size)
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    mask = mask.resize((new_w, new_h), resample=Image.NEAREST)  # mask 用最近邻
    # print('after resize:', image.size)
    # 计算 padding
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    right = target_size - new_w - left
    bottom = target_size - new_h - top
    pad_info = np.array([left, top, right, bottom])
    # 填充：图像用 0，mask 用 ignore_value
    image = T.Pad((left, top, right, bottom), fill=0)(image)
    # print('after padding:', image.size)
    # print('pad info', pad_info)
    mask = T.Pad((left, top, right, bottom), fill=ignore_value)(mask)

    return image, mask, pad_info

class Pseudo_Dataset(Dataset):
    def __init__(self, image_dir, pseudo_mask_dir, target_size=1022, is_train=False):
        train_ratio = 0.8
        self.image_dir = image_dir
        self.mask_dir = pseudo_mask_dir
        self.is_train = is_train
        # 检查图像和 mask 是否配对
        self.items = []
        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                mask_path = os.path.join(self.mask_dir, base + ".png")
                if os.path.exists(mask_path):
                    self.items.append((img_path, mask_path, base))
        
        shuffled_items = self.items.copy()
        random.shuffle(shuffled_items)
        # 划分 80% train, 20% val
        split_idx = int(len(shuffled_items) * train_ratio)
        if is_train:
            self.items = shuffled_items[:split_idx]
        else:
            self.items = shuffled_items[split_idx:]

        print(f"✅ 加载 {len(self.items)} 个图像-mask 对")
        # 图像标准化参数（DINOv2）
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], 
        )

        self.final_transform = v2.Compose([
            v2.ToTensor(),  # 确保是 TensorImage / PILImage
            v2.Normalize(mean=self.mean, std=self.std),
        ])


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, base_name = self.items[idx]

        # 读取图像和 mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度图，0-255
        # 将 mask 转为 0/1（假设 255=前景）
        mask_np_binary = (np.array(mask) > 128).astype(np.int32)  # 0 或 1
        mask_pil = Image.fromarray(mask_np_binary)

        # 同步 resize & pad（pad 区域设为 2）
        image_pil, mask_pil, pad_info = resize_and_pad(image, mask_pil, 1022, ignore_value=2)
        # 转为 numpy array
        mask_np = np.array(mask_pil)  # (1022, 1022), 值为 0, 1, 2
        # ⚡ 向量化生成 hard label
        labels = mask_to_hard_labels_vectorized(
            mask_np,
            patch_size=14,
            ignore_value=2,
            thresh_pos=0.3,
        )  # (Ph, Pw)
        # print('shape:', np.array(image_pil).shape, labels.shape)
        if self.is_train:
            augmented = self.transform(image=np.array(image_pil), mask=labels)
            image_pil = augmented['image']      # 增强后的 image
            labels = augmented['mask'] # 增强后的 mask（同步！）

        image_tensor = self.final_transform(image_pil)
        # print('after:',torch.unique(labels))
        downsampled_labels = cv2.resize(labels.astype(np.uint8), (0,0), fx=1/14, fy=1/14, interpolation=cv2.INTER_NEAREST)
        labels = torch.from_numpy(downsampled_labels).long()  # (Ph, Pw)
        # print('***'*5,image_tensor.shape, labels.shape,'***'*5)
        # print('pad info -----', pad_info)

        return image_tensor, labels, pad_info, base_name

class infer_Dataset(Dataset):
    def __init__(self, image_dir, target_size=1022):
        self.image_dir = image_dir
        # 检查图像和 mask 是否配对
        self.items = []
        for fname in sorted(os.listdir(image_dir)):
            if "NonCAM" in fname:
                break
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                self.items.append((img_path, base))

        print(f"✅ 加载 {len(self.items)} 个图像-mask 对")
        # 图像标准化参数（DINOv2）
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.final_transform = v2.Compose([
            v2.ToTensor(),  # 确保是 TensorImage / PILImage
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, base_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image_pil, _, pad_info = resize_and_pad(image, image, 1022, ignore_value=2)
        image_tensor = self.final_transform(image_pil)
        return image_tensor, pad_info, base_name, np.array(image.size)

class Sam_refine_Dataset(Dataset):
    def __init__(self, image_dir, pseudo_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = pseudo_mask_dir

        # 检查图像和 mask 是否配对
        self.items = []
        for fname in sorted(os.listdir(self.image_dir)):
            if "NonCAM" in fname:
                break
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                mask_path = os.path.join(self.mask_dir, base + ".jpg")
                if os.path.exists(mask_path):
                    self.items.append((img_path, mask_path, base))

        print(f"✅ 加载 {len(self.items)} 个图像-mask 对")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, base_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        mask = Image.open(mask_path).convert("L")  # 灰度图，0-255
        # 将 mask 转为 0/1（假设 255=前景）
        mask_np_binary = (np.array(mask) > 128).astype(np.int32)  # 0 或 1
        return image, mask_np_binary, base_name


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: (B, C, H, W) 未归一化的输出（logits）
        target: (B, H, W)      标签，可能包含 ignore_index
        """
        # 处理 ignore_index：将 ignore_index 位置 mask 掉
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            # 将 ignore_index 替换为 0（避免 one_hot 出错）
            target = target * valid_mask.long()
        else:
            valid_mask = None

        # 转为 one-hot
        num_classes = logits.shape[1]
        target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # softmax + 预测
        pred = F.softmax(logits, dim=1)

        # 应用 valid_mask
        if valid_mask is not None:
            pred = pred * valid_mask.unsqueeze(1)
            target = target * valid_mask.unsqueeze(1)

        # 计算 intersection 和 union
        intersection = (pred * target).sum(dim=(2, 3))  # (B, C)
        union = (pred + target - pred * target).sum(dim=(2, 3))  # (B, C)

        # IoU per class
        iou = (intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # 平均 IoU（忽略背景？可选）
        loss = 1.0 - iou  # IoU Loss = 1 - IoU
        loss = loss.mean()  # 可改为 weighted 或只算前景

        return loss


class DINOv2PatchClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='/data/yilong/hf_dinov2'):
        super().__init__()
        self.model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.dinov2 = AutoModel.from_pretrained(model_name)

        # 🔒 冻结 DINOv2 所有参数
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # 验证是否冻结成功
        print(f"Number of frozen parameters: {sum(p.numel() for p in self.dinov2.parameters() if not p.requires_grad)}")

        # 获取特征维度
        hidden_size = self.dinov2.config.hidden_size  # e.g., 768

        # 分类头（可训练）
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

        # 验证是否冻结成功
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)}")

    def forward(self, pixel_values):
        # 🔍 注意：no_grad 不在这里加，因为我们要获取梯度（只对 classifier）
        with torch.no_grad():  # ✅ 可选：用 no_grad 加速推理（但如果你要用梯度钩子，不要加）
            outputs = self.dinov2(pixel_values)
        # shape: [B, num_patches + 1, D]
        patch_features = outputs.last_hidden_state[:, 1:, :]  # 去掉 [CLS]

        H = W = 73
        patch_features = patch_features.view(-1, H, W, patch_features.shape[-1])  # [B, 73, 73, D]

        # 分类头（可训练）
        logits = self.classifier(patch_features)  # [B, 73, 73, num_classes]
        return logits.permute(0, 3, 1, 2)  # [B, num_classes, 73, 73]

def filter_bboxes(bboxes, iou_threshold=0.9):
    """
    过滤掉被其他 bbox 覆盖面积超过 90% 的 bbox
    Args:
        bboxes: list of [x1, y1, x2, y2]
        iou_threshold: 覆盖比例阈值（默认 0.9）
    Returns:
        list of 保留的 bboxes
    """
    def box_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def intersection_area(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0
        return (x_right - x_left) * (y_bottom - y_top)

    keep = []
    for i, box1 in enumerate(bboxes):
        area1 = box_area(box1)
        remove = False
        for j, box2 in enumerate(bboxes):
            if i == j:
                continue
            inter_area = intersection_area(box1, box2)
            # 覆盖比例是相对于 box1 自己的面积
            if area1 > 0 and (inter_area / area1) >= iou_threshold:
                remove = True
                break
        if not remove:
            keep.append(box1)
    return keep

def get_peaks(heatmap, bbox, threshold_abs=0.75, min_distance=56):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        x1 =0
        y1 =0
        x2 =heatmap.shape[0]
        y2 =heatmap.shape[1]
    # print(x1, y1, x2, y2)
    bbox_heatmap = heatmap[y1:y2, x1:x2]  # 提取 bbox 区域
    peaks = peak_local_max(
        bbox_heatmap,
        min_distance=min_distance,          # 波峰之间的最小像素距离
        threshold_abs=threshold_abs,       # 最低强度阈值（根据 heatmap 范围调整）
        exclude_border=True      # 允许在边界检测波峰
    )
    # 将 bbox 局部坐标转换为全局坐标
    global_peaks = peaks + np.array([y1, x1])  # 注意坐标顺序是 (y, x)
    global_peaks = np.flip(global_peaks, axis=1)
    return global_peaks

def expand_bbox(bbox, img_width=None, img_height=None, ratio=0.05):
    """
    拓展一个bbox
    
    参数:
        bbox: [x1, y1, x2, y2]
        img_width: 图像宽度（可选）
        img_height: 图像高度（可选）
        ratio: 扩展比例，默认0.05表示扩大5%

    返回:
        new_bbox: [new_x1, new_y1, new_x2, new_y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # 扩展量
    dw = int(w * ratio)
    dh = int(h * ratio)

    # 新bbox
    new_x1 = x1 - dw
    new_y1 = y1 - dh
    new_x2 = x2 + dw
    new_y2 = y2 + dh

    # 如果给定了图像大小，则裁剪
    # if img_width is not None:
    #     new_x1 = max(0, new_x1)
    #     new_x2 = min(img_width - 1, new_x2)
    # if img_height is not None:
    #     new_y1 = max(0, new_y1)
    #     new_y2 = min(img_height - 1, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]


def adaptive_dual_threshold_growth(heatmap):
    """
    基于高阈值+低阈值+区域生长的自适应二值化
    适用于兼顾大物体和小物体的 heatmap 分割
    参数:
        heatmap: 2D numpy 数组, 值越大表示目标可能性越高
        high_ratio: 高阈值占 heatmap 最大值的比例
        low_ratio:  低阈值占 heatmap 最大值的比例

    返回:
        final_mask: 0/1 mask, 保留大物体整体并分开小物体
        strong_mask: 高阈值核心区域
        weak_mask: 低阈值候选区域
    """
    # 归一化到 [0, 1]
    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    vals = hm_norm.ravel()
    low_ratio = 0.65
    high_ratio = 0.8
    # print(f'high: {high_ratio}, low:{low_ratio}, max: {hm_norm.max()}')
    # 高阈值和低阈值 mask
    strong_mask = hm_norm >= high_ratio
    weak_mask   = hm_norm >= low_ratio

    # 区域生长（dilation 方式）
    final_mask = morphology.reconstruction(
        seed=strong_mask.astype(np.uint8),
        mask=weak_mask.astype(np.uint8),
        method='dilation'
    )
    radius = 5  # 膨胀程度
    kernel = disk(radius)  # 圆形核
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    # 转为 0/1
    final_mask = final_mask.astype(np.uint8)    
    # rotated_boxes = get_rotated_box(final_mask*255)

    return final_mask, strong_mask.astype(np.uint8), weak_mask.astype(np.uint8)
 
def bbox_from_sim_maps(sim_maps):
    structure = generate_binary_structure(2, 2)
    all_boxes = []
    for heatmap in sim_maps:
        # 1. 阈值二值化（阈值可调节）
        # heatmap = heatmap-sim_maps_img
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)

        labeled_array, num_features = label(binary_map, structure=structure)
        # 3. 计算每个连通域中心坐标
        filtered_counts = []
        filtered_bboxes = []
        filtered_scores = []
        for idx in range(1, num_features + 1):  # 0是背景
            coords = np.column_stack(np.where(labeled_array == idx))
            count = len(coords)
            if count >= 14*14*3:  # 过滤小簇
                filtered_counts.append(count)
                # 计算bbox
                min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
                min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
                bbox = (min_x, min_y, max_x, max_y)
                filtered_bboxes.append(bbox)  # (x1, y1, x2, y2)
        
        # 过滤掉完全被大bbox覆盖的小bbox
        filtered_bboxes  = filter_bboxes(filtered_bboxes, iou_threshold=0.9)
        if len(filtered_bboxes)<20:
            all_boxes.append(filtered_bboxes)
        
        peaks = []
        neg_points = []

        for bbox in filtered_bboxes:
            x1, y1, x2, y2 = bbox
            peaks.append(get_peaks(heatmap, bbox, threshold_abs=0.8))
            new_bbox = expand_bbox(bbox, img_width=heatmap.shape[0], img_height=heatmap.shape[1])
            neg_points.append(get_peaks(1-heatmap, new_bbox, threshold_abs=0.95, min_distance=84))
            filtered_scores.append(heatmap[y1:y2, x1:x2].mean())

    all_boxes = [bbox for group in all_boxes for bbox in group]
    return all_boxes

from skimage import measure

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


def texture_scores(mask, image, P=8, R=1.0):
    """
    输入:
        mask: 0/1 二值mask
        image: 原图 (H, W, 3)
    输出:
        var_score: 方差对比分数
        lbp_score: LBP 熵对比分数
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    inside_vals = gray[mask > 0]
    outside_vals = gray[mask == 0]

    if len(inside_vals) == 0 or len(outside_vals) == 0:
        return 0.0, 0.0

    # --- 方法2: LBP 熵对比 ---
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    inside_entropy = shannon_entropy(lbp[mask > 0])
    outside_entropy = shannon_entropy(lbp[mask == 0])
    lbp_score = inside_entropy / (outside_entropy + 1e-6)
    return lbp_score


# 测试示例
def compare_masks(masks, image):
    results = []
    for i, m in enumerate(masks):
        var_s = texture_scores(m, image)
        results.append((i, var_s))
        # print(f"Mask {i}: VarScore={var_s:.3f}")
    return results


def binary_spatial_entropy_map(binary_image, window_size=3):
    """
    使用OpenCV进行更快的计算二值图像的局部空间熵图
    binary_image: 2D numpy array, dtype=bool 或 {0,1}
    window_size: 滑动窗口大小（建议奇数，如 3, 5）
    """
    structure = generate_binary_structure(2, 2)
    img = binary_image.astype(np.uint8)
    labeled_array, num_features = label(img, structure=structure)
    
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
    return entropy, num_features

def heatmap_spatial_entropy_map_fast(heatmap, window_size=9, bins=16):
    """
    高效计算 heatmap 的局部空间熵图
    heatmap: 2D numpy array (float)
    window_size: 滑动窗口大小（建议奇数）
    bins: 直方图bin数
    """
    h, w = heatmap.shape
    entropy_map = np.zeros((h, w), dtype=np.float32)

    heatmap = heatmap.astype(np.float32)
    # 归一化到 [0, bins-1]
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap_idx = np.floor(heatmap_norm * (bins - 1)).astype(np.uint8)

    total_pixels = window_size * window_size

    # 对每个 bin 统计局部直方图
    probs = np.zeros((h, w, bins), dtype=np.float32)
    for b in range(bins):
        mask = (heatmap_idx == b).astype(np.float32)
        count = cv2.boxFilter(mask, ddepth=-1, ksize=(window_size, window_size), 
                              normalize=False, borderType=cv2.BORDER_REFLECT)
        probs[:, :, b] = count / total_pixels

    # 计算熵
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log2(probs + 1e-12)
        entropy_map = -np.sum(probs * logp, axis=-1)

    return entropy_map

def clean_mask(mask, min_size=100):
    """清理掩码中的小孔洞和孤岛"""
    # 移除小的连通区域
    cleaned_mask = remove_small_objects(mask, min_size=min_size)
    # 填充小的孔洞
    cleaned_mask = fill_small_holes(cleaned_mask, area_threshold=min_size)
    return cleaned_mask

def remove_small_objects(mask, min_size=100):
    """移除小的连通区域"""
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    for region in regions:
        if region.area >= min_size:
            cleaned_mask[region.coords[:, 0], region.coords[:, 1]] = True
    return cleaned_mask

def fill_small_holes(mask, area_threshold=100):
    """填充小的孔洞"""
    inverted_mask = np.logical_not(mask)
    labeled_holes = measure.label(inverted_mask, connectivity=2)
    regions = measure.regionprops(labeled_holes)
    filled_mask = mask.copy()
    for region in regions:
        if region.area <= area_threshold:
            filled_mask[region.coords[:, 0], region.coords[:, 1]] = True
    return filled_mask


def calculate_iou_matrix(masks):
    """
    向量化计算前景IoU矩阵（更快）只考虑前景
    """
    n_masks = len(masks)
    if n_masks == 0:
        return np.array([])
    # 将所有mask转换为二维数组并展平
    masks_array = np.array([mask.astype(np.float32).flatten() for mask in masks])
    # 计算交集 (n_masks x n_masks)
    intersection = np.dot(masks_array, masks_array.T)
    # 计算每个mask的面积
    areas = np.sum(masks_array, axis=1)
    # 计算并集 (n_masks x n_masks)
    union = areas[:, None] + areas[None, :] - intersection
    # 计算IoU，避免除零错误
    iou_matrix = np.zeros((n_masks, n_masks))
    mask_nonzero = union > 0
    iou_matrix[mask_nonzero] = intersection[mask_nonzero] / union[mask_nonzero]
    # 对角线设为1（每个mask与自身的IoU）
    np.fill_diagonal(iou_matrix, 1.0)
    return iou_matrix


def calculate_spatial_chaos(mask):
    """
    计算mask的空间混乱程度
    混乱度越高，置信度越低
    """
    if np.sum(mask) == 0:
        return 0.0, 1.0  # 空mask，最高混乱度，最低置信度
    try:
        # 1. 计算空间熵
        entropy_map = heatmap_spatial_entropy_map_fast(mask, window_size=5)
        # 检查entropy_map是否有有效值
        if np.any(np.isnan(entropy_map)):
            entropy = 0.5  # 默认值
        else:
            # 分别计算前景和背景的熵，避免除零错误
            if np.any(mask == 1):
                entropy_foreground = np.mean(entropy_map[mask == 1])
            else:
                entropy_foreground = 0
                
            if np.any(mask == 0):
                entropy_background = np.mean(entropy_map[mask == 0])
            else:
                entropy_background = 0
                
            entropy = entropy_foreground + entropy_background
            entropy = min(max(entropy, 0), 1)  # 限制在0-1范围内

        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        # 2. 碎片化程度（连通区域越多越混乱）
        n_components = len(regions)
        fragmentation = min(n_components / 20.0, 1.0)  # 归一化
        # 综合混乱度（0-1范围，越高越混乱）
        chaos_score = 0.5 * fragmentation + 0.5 * entropy
        # 检查是否为NaN或异常值
        if np.isnan(chaos_score) or np.isinf(chaos_score):
            chaos_score = 0.5  # 默认值
        chaos_score = min(max(chaos_score, 0), 1)  # 限制在0-1范围内
        # 混乱度转换为置信度（混乱度越高，置信度越低）
        confidence = 1 - chaos_score
        # 最终检查
        if np.isnan(confidence) or np.isinf(confidence):
            confidence = 0.5
            chaos_score = 0.5
        return confidence, chaos_score
        
    except Exception as e:
        # 如果出现任何异常，返回默认值
        print(f"计算空间混乱度时出错: {e}")
        return 0.5, 0.5

def softmax(x):
    """
    计算list或numpy array的softmax
    """
    # 转换为numpy array
    x = np.array(x)
    # 减去最大值以提高数值稳定性（防止指数爆炸）
    x = x - np.max(x)
    # 计算softmax
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def calculate_all_confidences(masks):
    """
    计算所有mask的置信度
    """
    confidences = []
    chaos_scores = []
    
    for mask in masks:
        confidence, chaos = calculate_spatial_chaos(mask)
        confidences.append(confidence)
        chaos_scores.append(chaos)
    
    return np.array(confidences), np.array(chaos_scores)



def merge_masks_with_soft_voting(masks, heatmaps, image, iou_threshold=0.5, confidence_threshold=0.0):
    """
    使用soft voting合并mask
    """
    n_masks = len(masks)
    if n_masks == 0:
        return [], [], []
    
    # 计算IoU矩阵和置信度
    iou_matrix = calculate_iou_matrix(masks)
    confidences, chaos_scores = calculate_all_confidences(masks)
    confidences = softmax(confidences)
    # 使用并查集进行分组
    parent = list(range(n_masks))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            # 基于置信度决定合并方向（置信度高的作为根）
            if confidences[root_x] > confidences[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
    
    # 根据IoU阈值和置信度进行合并
    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            # 只有两个mask都有一定置信度且IoU足够高时才合并
            if (iou_matrix[i, j] > iou_threshold and 
                confidences[i] > confidence_threshold and 
                confidences[j] > confidence_threshold):
                union(i, j)
    
    # 分组
    groups = {}
    for i in range(n_masks):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # 对每个组进行soft voting合并
    merged_masks = []
    merge_groups = []
    group_score = []
    
    for group_indices in groups.values():
        if len(group_indices) == 0:
            continue
        # 使用soft voting创建合并后的mask
        merged_mask = soft_vote_merge(masks, heatmaps, image, group_indices, confidences)
        if merged_mask is None:
            continue
        merged_masks.append(merged_mask)
        merge_groups.append(group_indices)
        # group_score.append(score)
    
    return merged_masks, merge_groups, confidences, iou_matrix

def soft_vote_merge(masks, heatmaps, image, group_indices, confidences):
    """
    使用soft voting合并一组mask
    """
    if not group_indices:
        return np.zeros_like(masks[0], dtype=np.float32), 0.0
    
    H, W = masks[0].shape
    vote_map = np.zeros((H, W), dtype=np.float32)
    
    # 提取当前组的置信度并进行softmax归一化
    group_confidences = np.array([confidences[i] for i in group_indices])
    normalized_weights = softmax(group_confidences)
    total_confidence = 0.0
    
    # heatmap_entropy = 0
    for idx, weight in zip(group_indices, normalized_weights):
        # 使用softmax归一化后的权重作为投票权重
        # entropy_map = heatmap_spatial_entropy_map_fast(heatmaps[idx])
        # heatmap_entropy += entropy_map.mean()
        vote_map += masks[idx].astype(np.float32) * weight
        # vote_map += masks[idx].astype(np.float32) * heatmaps[idx]
        total_confidence += weight
    # 归一化投票结果（虽然权重已经是归一化的，但这里保持原逻辑）
    # heatmap_entropy /= len(group_indices)
    if total_confidence > 0:
        vote_map /= total_confidence
    # 二值化（阈值可调整）
    merged_mask = vote_map > 0.5
    cleaned_mask = clean_mask(merged_mask, min_size=100)

    if len(np.unique(cleaned_mask))==1:
        return None
    else:
        # score = scoring(cleaned_mask, image)
        # score.append(heatmap_entropy)
        # print(f'bc:{score[0]:.4f}, entropy:{score[1]:.4f}, ts: {score[2]:.4f}, h_entropy: {score[3]:.4f}')
        return cleaned_mask

def scoring(mask, image):
    bc = boundary_contact(mask, n=10)
    entropy_map, num_comp = binary_spatial_entropy_map(mask, window_size=7)
    ts = texture_scores(mask, image)
    # entropy = np.mean(entropy_map[mask == 1])+np.mean(entropy_map[mask == 0])
    entropy = np.mean(entropy_map[mask == 1])
    # res = rotation_invariant_lbp_boundary_contrast(image, mask, P=8, R=1.0, inner_margin=3, outer_margin=6, metric="js")
    score = [bc,entropy,ts]
    return score      
# 可视化函数

def visualize_merging_process(original_masks, image, merged_masks, merge_groups, confidences):
    """
    可视化合并过程
    """
    structure = generate_binary_structure(2, 2)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    confidences = softmax(confidences)
    
    n_original = len(original_masks)
    n_merged = len(merged_masks)
    
    fig, axes = plt.subplots(math.ceil((n_merged+n_original+1)/6), 6, figsize=(24, 15))
    axes = axes.flatten()
    # 显示原始mask和置信度
    axes[0].imshow(image)
    for i in range(1, n_original+n_merged+1):
        if i < n_original+1:
            axes[i].imshow(original_masks[i-1], cmap='jet')
            labeled_array, num_features = label(original_masks[i-1], structure=structure)
            axes[i].set_title(f'Mask {i-1}, {num_features} comps')
        else:
            axes[i].imshow(merged_masks[i-n_original-1], cmap='jet')
            labeled_array, num_features = label(merged_masks[i-n_original-1], structure=structure)
            group_str = ','.join(map(str, merge_groups[i-n_original-1]))
            axes[i].set_title(f'Group {i-n_original-1}\n({group_str}, {num_features} comps)')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_iou_matrix(boxes):
    boxes = np.array(boxes)
    x1 = boxes[:, 0][:, None]
    y1 = boxes[:, 1][:, None]
    x2 = boxes[:, 2][:, None]
    y2 = boxes[:, 3][:, None]

    xx1 = np.maximum(x1, x1.T)
    yy1 = np.maximum(y1, y1.T)
    xx2 = np.minimum(x2, x2.T)
    yy2 = np.minimum(y2, y2.T)

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    intersection = w * h

    area = (x2 - x1) * (y2 - y1)
    union = area + area.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    return iou_matrix

def merge_boxes_iterative(boxes, iou_thresh=0.9):
    boxes = np.array(boxes)
    while True:
        iou_matrix = compute_iou_matrix(boxes)
        np.fill_diagonal(iou_matrix, 0)
        if np.max(iou_matrix) <= iou_thresh:
            break
        # 构建 IoU > 阈值的邻接矩阵
        adj = (iou_matrix > iou_thresh).astype(int)
        # 找连通分量
        graph = csr_matrix(adj)
        n_components, labels = connected_components(csgraph=graph, directed=False)
        # 对每个连通分量合并 bbox
        new_boxes = []
        for i in range(n_components):
            group = boxes[labels == i]
            x1 = group[:,0].min()
            y1 = group[:,1].min()
            x2 = group[:,2].max()
            y2 = group[:,3].max()
            new_boxes.append([x1, y1, x2, y2])
        boxes = np.array(new_boxes)
    return boxes

def draw_bboxes_watershed(heatmap, threshold_ratio=0.6, min_area=42, show=True):
    """
    使用分水岭算法从 heatmap 中分离物体并绘制外接矩形

    参数:
        heatmap : np.ndarray
            输入单通道热力图 (float 或 uint8)
        threshold_ratio : float
            阈值比例，例如 0.6 表示阈值 = 0.6 * max(heatmap)
        min_area : int
            过滤过小的区域 (像素数小于该值的区域会被忽略)
        show : bool
            是否显示结果图像

    返回:
        img_with_boxes : np.ndarray
            带框的图像
        bboxes : list of tuple
            每个框的坐标 (x, y, w, h)
    """
    # 归一化到 0~255
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 阈值化
    thresh_val = int(threshold_ratio * heatmap_norm.max())
    _, binary = cv2.threshold(heatmap_norm, int(thresh_val), 255, cv2.THRESH_BINARY)

    # 去噪：形态学开运算
    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 距离变换找到前景
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 转换到三通道，应用分水岭
    img_color = cv2.cvtColor(heatmap_norm, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 绘制外接矩形
    img_with_boxes = img_color.copy()
    bboxes = []
    for label in range(2, num_labels+1):  # label=0是未知，1是背景，从2开始
        mask = (markers == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < min_area:  # 过滤小区域
                continue
            bboxes.append((x, y, x+w, y+h))
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0,0,255), 2)
    # 可视化
    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Watershed Result")
        plt.axis("off")
        plt.show()

    return img_with_boxes, bboxes
    
def sigmoid(x):
    low = -10
    high= 10
    x = low + (x - x.min()) * (high - low) / (x.max() - x.min())
    return 1 / (1 + np.exp(-x)) 

def get_candidate_mask_from_SAM(sim_maps, image, predictor):
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    predictions =  []
    binary_maps = []
    for index, heatmap in enumerate(sim_maps):
        # 1. 阈值二值化（阈值可调节）
        # heatmap = heatmap-sim_maps_img
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0) 
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
        
        binary_maps.append(binary_map)
        labeled_array, num_features = label(binary_map, structure=structure)
        # 3. 计算每个连通域中心坐标
        filtered_bboxes = []
        img_size = heatmap.shape[0]*heatmap.shape[1]
        for idx in range(1, num_features + 1):  # 0是背景
            coords = np.column_stack(np.where(labeled_array == idx))
            # 计算bbox
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)
            if (max_x-min_x)*(max_y-min_y) >= heatmap.shape[0]*heatmap.shape[0]*0.001:  # 过滤小簇
                filtered_bboxes.append(bbox)  # (x1, y1, x2, y2)
        
        # 过滤掉完全被大bbox覆盖的小bbox
        if len(filtered_bboxes)>=1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)

        if len(filtered_bboxes) <= 12 and len(filtered_bboxes) > 0:
            PRED = np.empty([0, image.shape[0], image.shape[1]])
            for bbox in filtered_bboxes:
                if (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) > 0.4 * heatmap.shape[0] * heatmap.shape[1]:
                    mask, iou_scores, low_res_masks = predictor.predict(
                            box = bbox, 
                            mask_input = cv2.resize(heatmap, dsize=(256,256))[None,...],
                            multimask_output=False
                    )
                else:
                    mask, iou_scores, low_res_masks = predictor.predict(
                            box = bbox, 
                            mask_input = cv2.resize(heatmap, dsize=(256,256))[None,...],
                            # mask_input = cv2.resize(binary_map, dsize=(256,256))[None,...],
                            multimask_output=False
                    )
                PRED = np.concatenate([PRED, mask],axis=0)
            combined_mask = np.any(PRED, axis=0)
            mask = combined_mask
            mask = clean_mask(mask, min_size=100)
            labeled_array, num_features = label(mask, structure=structure)
            bcr = boundary_contact(mask, n=10)
            if bcr>0.75:
                continue
            predictions.append(mask)
    return predictions, binary_maps

def top_k_masks(masks, scores, k=3):
    """
    根据得分选出最高的 k 个候选 mask
    masks: list，每个元素是一个 mask (numpy array)
    scores: list or numpy array，对应每个 mask 的得分
    k: int，返回的数量
    """
    scores = np.array(scores)
    # 得分从大到小排序，取前 k 个索引
    topk_indices = scores.argsort()[::-1][:k]
    # 根据索引取出对应的 mask
    topk_masks = [masks[i] for i in topk_indices]
    topk_scores = [scores[i] for i in topk_indices]

    return topk_masks, topk_scores

def get_MAX_IoU_mask_from_SAM(sim_maps, image, predictor, k=3):
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    all_boxes = []
    predictions =  []
    binary_maps = []
    scores = []
    for index, heatmap in enumerate(sim_maps):
        # 1. 阈值二值化（阈值可调节）
        # heatmap = heatmap-sim_maps_img
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0) 
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
        
        binary_maps.append(binary_map)
        labeled_array, num_features = label(binary_map, structure=structure)
        # 3. 计算每个连通域中心坐标
        filtered_counts = []
        filtered_bboxes = []
        img_size = heatmap.shape[0]*heatmap.shape[1]
        for idx in range(1, num_features + 1):  # 0是背景
            coords = np.column_stack(np.where(labeled_array == idx))
            count = len(coords)

            filtered_counts.append(count)
            # 计算bbox
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= heatmap.shape[0]*heatmap.shape[0]*0.01:  # 过滤小簇
                filtered_bboxes.append(bbox)  # (x1, y1, x2, y2)
        
        if len(filtered_bboxes)>=1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            PRED = np.empty([0, image.shape[0], image.shape[1]])
            mask, iou_scores, low_res_masks = predictor.predict(
                    box = filtered_bboxes, 
                    mask_input = cv2.resize(heatmap, dsize=(256,256))[None,...],
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape)==3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr>0.75:
                mask = 1-mask
                heatmap = 1-heatmap

            mask = clean_mask(mask, min_size=100)
            if mask.sum()==0:
                continue
            _, num_features = label(mask, structure=structure)
            corr, p_val = pearsonr(heatmap.flatten(), mask.flatten())
            bc = 1-boundary_contact(mask, n=10)
            score = corr+bc
            scores.append(score)
            predictions.append(mask)     
    topk_preds, topk_scores = top_k_masks(predictions, scores, k=k) 
    # return topk_preds, topk_scores
    return predictions, topk_scores


def get_MAX_IoU_mask_from_SAM_bbox_from_binary_map(refined_masks, candidate_fgs, leiden_map, image, predictor):
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    all_boxes = []
    predictions =  []
    binary_maps = []
    scores = []
    for refined_mask in refined_masks:
        labeled_array_1, num_features_1 = label(refined_mask, structure=structure)
        filtered_counts = []
        filtered_bboxes = []

        for idx in range(1, num_features_1+1):  # 0是背景
            coords = np.column_stack(np.where(labeled_array_1 == idx))
            count = len(coords)

            filtered_counts.append(count)
            # 计算bbox
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= image.shape[0]*image.shape[0]*0.01:  # 过滤小簇
                filtered_bboxes.append(bbox)  
        
        if len(filtered_bboxes)>=1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            PRED = np.empty([0, image.shape[0], image.shape[1]])
            mask, iou_scores, low_res_masks = predictor.predict(
                    box = filtered_bboxes, 
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape)==3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr>0.75:
                mask = 1-mask

            mask = clean_mask(mask, min_size=100)
            if mask.sum()==0:
                continue
            _, num_features = label(mask, structure=structure)
            predictions.append(mask)  
    # print(f'num of predictions from refined masks: {len(predictions)}, unique fg: {np.unique(leiden_map)}')

    for candidate_fg in np.unique(leiden_map):
        labeled_array_2, num_features_2 = label(leiden_map==candidate_fg, structure=structure)
        filtered_counts = []
        filtered_bboxes = []
        for idx in range(1, num_features_2+1):  # 0是背景
            coords = np.column_stack(np.where(labeled_array_2 == idx))
            count = len(coords)

            filtered_counts.append(count)
            # 计算bbox
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= image.shape[0]*image.shape[0]*0.01:  # 过滤小簇
                filtered_bboxes.append(bbox)  # (x1, y1, x2, y2)
        
        if len(filtered_bboxes)>=1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            PRED = np.empty([0, image.shape[0], image.shape[1]])
            mask, iou_scores, low_res_masks = predictor.predict(
                    box = filtered_bboxes, 
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape)==3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr>0.75:
                mask = 1-mask

            mask = clean_mask(mask, min_size=100)
            if mask.sum()==0:
                continue
            _, num_features = label(mask, structure=structure)
            predictions.append(mask)  
    # print(f'num of predictions from leiden maps: {len(predictions)-len(refined_masks)}')

    return predictions, None


def get_one_candidate_mask_from_SAM(sim_map, image, Candidate_mask, predictor):
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    heatmap = cv2.GaussianBlur(sim_map, (15, 15), 0) 
    heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
    labeled_array, num_features = label(binary_map, structure=structure)
    # 3. 计算每个连通域中心坐标
    filtered_bboxes = []
    for idx in range(1, num_features + 1):  # 0是背景
        coords = np.column_stack(np.where(labeled_array == idx))
        count = len(coords)

        if count >= 14*14*3:  # 过滤小簇
            # 计算bbox
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)
            filtered_bboxes.append(bbox)  # (x1, y1, x2, y2)
    
    boundary_contact_ratio = boundary_contact(Candidate_mask)
    _, watershed_bboxes = draw_bboxes_watershed(sim_map, threshold_ratio=0.55, min_area=28, show=False)
    filtered_bboxes.extend(watershed_bboxes)
    filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
    filtered_bboxes = np.array(filtered_bboxes)
    if len(filtered_bboxes)>20:
        return None
    mask, _, _ = predictor.predict(
            box = filtered_bboxes, 
            mask_input = cv2.resize(heatmap, dsize=(256,256))[None,...],
            multimask_output=False
    )
    if len(mask.shape) == 3:
        mask = mask[None,...]
    combined_mask = np.any(mask, axis=0)[0]
    return combined_mask


def save_mask_as_color(mask, save_path, colormap='jet'):
    """
    mask: numpy array, shape (H, W) or (H, W, 1)
    save_path: str, path to save the color mask
    colormap: str, matplotlib colormap name
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    # get colormap
    '''
    cmap = plt.get_cmap(colormap)
    mask_color = (cmap(mask)[:, :, :3] * 255).astype(np.uint8)  # 去掉alpha
    Image.fromarray(mask_color).save(save_path)
    '''
    mask = mask.astype(np.uint8)
    Image.fromarray(mask*255).save(save_path)


def sample_heatmap_points(heatmap, step=24, high_thresh=0.7, low_thresh=0.3):
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
    H, W = heatmap.shape
    coords = []
    labels = []
    # 按网格步长采样
    for y in range(0, H, step):
        for x in range(0, W, step):
            val = heatmap[y,x]
            if val > high_thresh:
                coords.append((x, y))
                labels.append(1)
            elif val < low_thresh:
                coords.append((x, y))
                labels.append(0)
    return np.array(coords), np.array(labels)

def sample_heatmap_points_with_bboxes(heatmap, bboxes, step=42, high_thresh=0.8, low_thresh=0.3):
    """
    稀疏采样 heatmap，并结合 bboxes 进行筛选：
      - bbox 内：只保留 > high_thresh 的点，标签=1
      - bbox 外：只保留 < low_thresh 的点，标签=0
      - 其他情况直接丢弃
    参数：
        heatmap (ndarray): (H, W) 值范围 [0, 1]
        bboxes (list/ndarray): [(x_min, y_min, x_max, y_max), ...]  基于 heatmap 像素坐标
        step (int): 采样间隔
        high_thresh (float): 高阈值
        low_thresh (float): 低阈值
    返回：
        coords (ndarray): shape = (k, 2)，每行是 (y, x) 坐标
        labels (list[int]): 对应的标签（0 或 1）
    """
    H, W = heatmap.shape
    coords = []
    labels = []

    def in_any_bbox(x, y):
        for (x1, y1, x2, y2) in bboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    for y in range(0, H, step):
        for x in range(0, W, step):
            val = heatmap[y, x]
            inside = in_any_bbox(x, y)

            if inside:
                if val > high_thresh:
                    coords.append((x, y))
                    labels.append(1)
                elif val < low_thresh:
                    coords.append((x, y))
                    labels.append(0)
            # elif (not inside) and val < low_thresh:
            #     coords.append((y, x))
            #     labels.append(0)
            # 其他情况直接丢弃

    return np.array(coords), np.array(labels)

def mask_sim(list_of_masks):
    flattened_masks = np.array([mask.ravel() for mask in list_of_masks])
    cosine_sim_matrix = cosine_similarity(flattened_masks)
    return cosine_sim_matrix
    
def sim_of_sim_maps(sim_maps):
    M = mask_sim(sim_maps) 
    plt.figure(figsize=(10, 10))
    plt.imshow(M, cmap='jet')
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            text_color = 'white' if M[i, j] < 0.5 else 'black'
            _ = plt.text(j, i, f'{M[i, j]:.3f}', 
                    ha='center', va='center', 
                    color=text_color, 
                    fontsize=10) # 可以调整字体大小
    plt.show()
    return M


def strict_components_from_similarity(M, thresh=0.5):
    """
    输入:
        M: 相似度矩阵 (numpy array, n x n)
        thresh: 阈值，大于这个数值的认为可以合并
    输出:
        comps: 列表，每个元素是一个组合(列表)，里面是满足条件的索引集合
    规则:
        只有当组合里所有两两相似度都 > thresh，才算同一个组合
    """
    n = M.shape[0]
    comps = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        comp = [i]
        for j in range(n):
            if j != i and j not in used:
                # 检查 j 和 comp 中所有元素的相似度是否都大于 thresh
                if all(M[j, k] > thresh for k in comp):
                    comp.append(j)
        # 标记已用
        for k in comp:
            used.add(k)
        comps.append(comp)

    return comps

def heatmap_correlation_matrix(heatmaps, show=False):
    """
    计算一组 heatmap 之间的相关系数矩阵（向量化实现）

    参数
    ----
    heatmaps: list of numpy.ndarray
        每个元素是一张 heatmap (2D 数组)

    返回
    ----
    corr_matrix: numpy.ndarray
        相关系数矩阵 (n x n)，对称矩阵，对角线为 1
    """
    # 将所有 heatmap 拉平并堆叠成 2D 数组: shape (n, d)
    flat_maps = np.array([hm.flatten() for hm in heatmaps])  
    # np.corrcoef 要求输入是 (features, samples)，所以转置
    corr_matrix = np.corrcoef(flat_maps)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(corr_matrix, cmap='jet')
        rows, cols = corr_matrix.shape
        for i in range(rows):
            for j in range(cols):
                text_color = 'white' if corr_matrix[i, j] < 0.5 else 'black'
                _ = plt.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                        ha='center', va='center', 
                        color=text_color, 
                        fontsize=10) # 可以调整字体大小
        plt.show()
    return corr_matrix


def merge_sim_maps(sim_maps, thresh=0.89, visualisation=False):
    # M = sim_of_sim_maps(sim_maps)
    M = heatmap_correlation_matrix(sim_maps,show=visualisation) 
    pairs = strict_components_from_similarity(M, thresh=thresh)
    # print(f'pairs:{pairs}')
    merged_sim_maps = []
    if visualisation:
        fig, axs = plt.subplots(1, len(pairs), figsize=(15,8))
        axs = axs.flatten()
        for ax,pair in zip(axs, pairs):
            maps_to_merge = [sim_maps[idx] for idx in pair]
            # 沿第一个轴取最大值
            max_map = np.max(maps_to_merge, axis=0)
            merged_sim_maps.append(max_map)
            # print(f'-----:{(maps!=max_map).sum()}')
            _ = ax.imshow(max_map, cmap='jet')  # 可视化子矩阵本身
            _ = ax.set_title(f"ids: {pair}")
        plt.show()
    else:
        for pair in pairs:
            maps_to_merge = [sim_maps[idx] for idx in pair]
            # 沿第一个轴取最大值
            max_map = np.max(maps_to_merge, axis=0)
            merged_sim_maps.append(max_map)
            # print(f'-----:{(maps!=max_map).sum()}')
    return merged_sim_maps  

def bboxes_to_mask(bboxes, img_size):
    """
    将一组 bboxes 转换为 mask
    bboxes: [[x1,y1,x2,y2], ...]
    img_size: (h, w)
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return mask

def compute_iou(mask1, mask2):
    """
    计算两个 mask 的 IoU
    """
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union

def find_best_mask_by_iou(mask_list, gt_mask):
    """
    从mask列表中找到与gt_mask的IoU最高的mask
    Args:
        mask_list: list of numpy arrays, 每个都是二值mask (0和1)
        gt_mask: numpy array, 真实的二值mask (0和1)
    Returns:
        best_mask: 与gt_mask IoU最高的mask
        best_iou: 最高的IoU值
        best_index: 最佳mask在列表中的索引
        all_ious: 所有mask的IoU值列表
    """
    # 确保mask是二值的
    gt_mask = (gt_mask > 0).astype(np.uint8)
    best_iou = -1
    best_mask = None
    best_index = -1
    all_ious = []
    for i, candidate_mask in enumerate(mask_list):
        # 确保候选mask是二值的
        candidate_mask_binary = (candidate_mask > 0).astype(np.uint8)
        # 计算IoU
        iou = compute_iou(candidate_mask_binary, gt_mask)
        all_ious.append(iou)
        # 更新最佳mask
        if iou > best_iou:
            best_iou = iou
            best_mask = candidate_mask_binary
            best_index = i
    return best_mask, best_iou, best_index, all_ious

def find_best_match_box(bboxes_ref, candidates, img_size):
    """
    找到和 bboxes_ref 重合度最大的 bboxes 组

    参数:
        bboxes_ref: 参考框集合 [[x1,y1,x2,y2], ...]
        candidates: [bboxes_1, bboxes_2, ...]  每一组都是一个 bbox 列表
        img_size: (h, w) 图像尺寸

    返回:
        best_idx: 最优集合索引
        best_iou: 最大 IoU
    """
    mask_ref = bboxes_to_mask(bboxes_ref, img_size)
    best_idx, best_iou = -1, 0.0
    scores = []
    for i, bboxes in enumerate(candidates):
        mask_cand = bboxes_to_mask(bboxes, img_size)
        iou = compute_iou(mask_ref, mask_cand)
        scores.append(iou)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou, scores

def resize_bbox(bbox, orig_size, new_size):
    """
    将原图的 bbox 映射到新图坐标系
    参数:
        bbox: [x1, y1, x2, y2] (基于原图的坐标)
        orig_size: (w, h) 原图尺寸
        new_size: (new_w, new_h) 新图尺寸
    返回:
        new_bbox: [new_x1, new_y1, new_x2, new_y2] (基于新图的坐标)
    """
    x1, y1, x2, y2 = bbox
    w, h = orig_size
    new_w, new_h = new_size
    scale_x = new_w / w
    scale_y = new_h / h
    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y
    return [new_x1, new_y1, new_x2, new_y2]
    
def resize_bboxes(bboxes: list, original_size: tuple, new_size: tuple) -> list:
    return [resize_bbox(bbox, original_size, new_size) for bbox in bboxes]

def numpy_to_base64(img_array):
    """将numpy数组转换为base64编码的PNG图像"""
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_json_from_markdown(markdown_string):
    """
    从Markdown格式的字符串中提取并解析JSON内容
    
    参数:
    markdown_string: 包含JSON的Markdown字符串
    
    返回:
    dict or list: 解析后的JSON对象
    """
    try:
        # 方法1: 使用正则表达式移除Markdown代码块标记
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, markdown_string, re.DOTALL)
        
        if match:
            # 提取JSON部分
            json_content = match.group(1).strip()
            # 解析JSON
            return json.loads(json_content)
        else:
            # 如果没有找到代码块标记，尝试直接解析
            return json.loads(markdown_string.strip())
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None

def combine_images_cv2_advanced(images_dict, spacing=10, background_color=255, paths=None):
    """
    高级版图片拼接，支持字典输入，在每个mask下方显示key
    
    Args:
        images_dict: 图片字典 {key: mask_image}
        spacing: 图片间距（像素）
        background_color: 背景颜色 (灰度值)
        color_image: 彩色图片 (BGR格式)
    """
    
    n = len(images_dict)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # 获取第一个mask的尺寸作为基准
    first_mask = list(images_dict.values())[0]
    h, w = first_mask.shape[:2]

    
    # 计算包含间距和文字区域的总尺寸
    total_width = cols * w + (cols - 1) * spacing
    total_height = rows * h + (rows - 1) * spacing
    
    # 创建新图片（灰度背景）
    new_image = np.ones((total_height, total_width, 3), dtype=np.uint8)
    new_image[:, :] = (background_color, background_color, background_color)
    
    x_start = 0  # 所有mask左侧对齐彩色图像    
    for i, (key, mask) in enumerate(images_dict.items()):
        row = i // cols
        col = i % cols
        
        # 计算位置（包含间距）
        y_start = row * (h + spacing)
        
        # 计算居中偏移量
        x_offset = col * (w + spacing)  # 列偏移
        y_offset = 0  # 不需要垂直偏移
        
        y1 = y_start + y_offset
        y2 = y1 + mask.shape[0]
        x1 = x_start + x_offset
        x2 = x1 + mask.shape[1]

        
        # 确保mask是2D的（灰度图）
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        if len(mask.shape) == 3:
            mask_bgr = mask
        else:
            mask_gray = mask    
            # 将灰度图转为3通道以便放入彩色背景
            mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        
        # 确保不越界
        if y2 <= new_image.shape[0] and x2 <= new_image.shape[1]:
            new_image[y1:y2, x1:x2] = mask_bgr
        else:
            print(f"警告: 图片 {key} 超出边界")
    
    # cv2.imwrite(f"temp_mask/{paths}", new_image)
    return new_image

def clean_and_parse_json(text):
    """
    解析大模型输出的文本，提取JSON格式的字典
    
    Args:
        text: 包含JSON数据的字符串或字符串列表
    
    Returns:
        list: 解析后的JSON数据列表
    """
    # 如果输入是列表，转换为字符串
    if isinstance(text, list):
        text = ' '.join(text)
    
    # 方法1：尝试直接解析整个文本
    try:
        # 去除可能的多余空格和换行
        cleaned_text = text.strip()
        # 尝试直接解析为JSON
        result = json.loads(cleaned_text)
        return result
    except json.JSONDecodeError:
        pass
    
    # 方法2：使用正则表达式提取JSON部分
    try:
        # 匹配 ```json 和 ``` 之间的内容
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            result = json.loads(json_str)
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # 方法3：尝试提取最外层的中括号内容
    try:
        # 匹配最外层的数组
        array_pattern = r'\[.*\]'
        match = re.search(array_pattern, text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # 如果所有方法都失败，抛出异常
    print(text)
    raise ValueError("无法从文本中解析出有效的JSON数据")

def get_pred_from_QWen(img_mask_grid, new_masks, QWen_model, QWen_processor, device="cuda:1"):
    mask_b64 = numpy_to_base64(img_mask_grid)
    # img_b64 = numpy_to_base64(cv2_img)
    messages = [
        {
        "role": "user",
        "content": [
            # { "type": "image", "image" : f"data:image/png;base64,{img_b64}"},
            { "type": "image", "image" : f"data:image/png;base64,{mask_b64}"},
            {"type": "text", "text": f"""CAMOUFLAGE MASK SELECTION TASK
                IMAGE: Contains camouflaged objects that blend with surroundings.
                MASKS: Candidate masks with bounding boxes.

                MASK INTERPRETATION:
                - White areas = potential hidden/concealed objects
                - Black areas = background  
                - Boxes = object location hints

                CRITICAL REQUIREMENT: The chosen mask MUST match your object analysis.

                STEP-BY-STEP PROCESS:

                1. OBJECT IDENTIFICATION:
                - Carefully examine the image
                - Identify all hidden/concealed/camouflaged objects and their exact locations
                

                2. MASK EVALUATION (for each mask):
                Mask X: Does the white region cover all identified objects? How much extra background?
                Mask Y: Does the white region cover all identified objects? How much extra background?

                3. SELECTION CRITERIA:
                - PRIMARY: Choose the mask that covers ALL identified objects completely
                - SECONDARY: Among masks that meet primary criterion, choose the one with least background
                - If no mask covers all objects, choose the one that covers the most objects

                4. CONSISTENCY CHECK:
                - Ensure the chosen mask's white regions align with your identified objects
                - If analysis says "2 camouflaged insects", the mask should have 2 corresponding white regions

                OUTPUT JSON (DO NOT ADD ANY EXTRA INFO, JUST JSON!):
                [{{
                    "best_mask": "Mask X" [Mask X should be one of the candidate masks], 
                }}]
            """ }
        ]}
    ]
    text = QWen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = QWen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    # Inference: Generation of the output
    generated_ids = QWen_model.generate(**inputs, 
        max_new_tokens=512, 
        output_scores=True, 
        do_sample=True,       # 一定要开启 sampling
        temperature=0.2, 
        return_dict_in_generate=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
    ]
    output_text = QWen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    try:
        data = clean_and_parse_json(output_text)
        if isinstance(data, list):
            data = clean_and_parse_json(output_text)[0]
    except:
        data = clean_and_parse_json(output_text)
        print(output_text)
    # pprint(data)
    # data = json.loads(json_text)[0]
    try:
        selected_mask = data["best_mask"].split(' ')[1]
    except:
        # print(data)
        selected_mask = None
    # print(f'selected mask: {selected_mask}, keys: {new_masks.keys()}')
    # final_pred = new_masks[selected_mask]
    return selected_mask
