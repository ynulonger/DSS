import os
import json
import torch
import random
import argparse
import numpy as np
from utils import *
from glob import glob
from PIL import Image
from tqdm import tqdm
from ntpath import exists
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from refine_leiden_utils import *
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor



def get_img_list(json_file, pred_dir):
    pred_list = glob(os.path.join(pred_dir, "*.png"))
    pred_list = [os.path.basename(pred).replace(".png", "") for pred in pred_list]
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    img_list = []
    bbox_list = []
    print(f'{len(pred_list)} imgs finished, left {len(results)-len(pred_list)} imgs to process')

    for item in tqdm(results):
        confidence = []
        img_path = item["image"]
        img_name = img_path.split("/")[-1].split(".")[0]
        if img_name in pred_list:
            continue
        result = item["result"]
        # 兼容result为dict或list
        if isinstance(result, str):
            continue
        elif isinstance(result, list):
            bboxes = []
            for obj in result:
                if isinstance(obj, dict) and "confidence" in obj:
                    confidence.append(obj["confidence"])
                if isinstance(obj, dict) and "bbox_2d" in obj:
                    bboxes.append(obj["bbox_2d"])
            # print(bboxes)
        elif isinstance(result, dict):
            bboxes = []
            if "description" in result.keys():
                img = cv2.imread(img_path, flags=0)
                bboxes.append([0, img.shape[0], 0, img.shape[1]])
                # continue
            for obj in result:
                if obj == "confidence":
                    confidence.append(result["confidence"])
                if obj == "bbox_2d":
                    bboxes.append(result["bbox_2d"])

        bboxes_np = np.array(bboxes)

        if len(bboxes_np.shape)==3:
            print(img_path, bboxes_np)
            bboxes_np = bboxes_np[0]
        # if np.array(confidence).mean()>confidence_threshold:
        bbox_list.append(bboxes_np)
        img_list.append(img_path)

    print(f'num of images: {len(img_list)}, {len(bbox_list)}')
    return img_list, bbox_list

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

from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor

def pairwise_selection_from_QWen(image, mask_list, score_list, QWen_model, QWen_processor, device="cuda:0"):
    """
    基于得分排序，通过两两比较选择最佳mask
    
    Args:
        image: 原始图像
        mask_list: mask列表
        score_list: 对应的得分列表（得分越低越好）
        QWen_model: QWen模型
        QWen_processor: QWen处理器
        device: 设备
    """
    # 创建mask和得分的配对列表
    mask_score_pairs = list(zip(mask_list, score_list))
    
    # 按得分升序排序（得分越低越好）
    mask_score_pairs.sort(key=lambda x: x[1])
    
    # 如果只有一个mask，直接返回
    if len(mask_score_pairs) == 1:
        return mask_score_pairs[0][0]
    
    # 使用迭代方式进行两两比较
    current_list = mask_score_pairs.copy()
    
    while len(current_list) > 1:
        # 选择得分最低的两个mask进行比较
        mask1, score1 = current_list[0]
        mask2, score2 = current_list[1]
        remaining_masks = current_list[2:]
        # print('mask1 shape:', mask1.shape, 'score1:', 'img shape:', image.shape)
        masked_img1 = mask1[:, :, np.newaxis]*image
        masked_img2 = mask2[:, :, np.newaxis]*image
        # 调用QWen进行选择
        votes = []
        for i in range(1):
            selected_mask_name = query_qwen_for_selection(image, masked_img1, masked_img2, QWen_model, QWen_processor, device)
            votes.append(selected_mask_name.lower())
        
        vote_count = Counter(votes)
        # print(f"投票结果: {dict(vote_count)}")
        # 返回得票最多的mask
        selected_mask_name = vote_count.most_common(1)[0][0]
        # print("Vote Count:", vote_count)

        if selected_mask_name.lower() == "mask a":
            winner = (mask1, score1)
        elif selected_mask_name.lower() == "mask b":
            winner = (mask2, score2)
            
        '''     
        selected_mask_name = query_qwen_for_selection(image, masked_img1, masked_img2, QWen_model, QWen_processor, device)
        print("Selected Mask:", selected_mask_name)
        # 根据选择结果确定胜出的mask
        if selected_mask_name.lower() == "mask a":
            winner = (mask1, score1)
        elif selected_mask_name.lower() == "mask b":
            winner = (mask2, score2)
        else:
            # 如果无法确定，默认选择得分更高的
            winner = (mask2, score2)
            print("无法确定选择，默认选择得分更高的mask")
        '''
        '''
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(masked_img1)
        axs[1].set_title("Mask A: Score {:.4f}".format(score1))
        axs[2].imshow(masked_img2)
        axs[2].set_title("Mask B: Score {:.4f}".format(score2))
        axs[3].imshow(winner[0][:, :, np.newaxis]*image)
        axs[3].set_title("Winner Mask: Score {:.4f}".format(winner[1]))
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        '''
        # 将胜出者放回剩余列表中，保持排序
        new_list = [winner] + remaining_masks
        new_list.sort(key=lambda x: x[1])
        current_list = new_list.copy()
    
    # 返回最终胜出的mask
    return current_list[0][0]

def resize_for_faster_processing(img, max_size=1024):
    """降低图像分辨率以加速处理"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        import cv2
        return cv2.resize(img, (new_w, new_h))
    return img

def query_qwen_for_selection(image, mask1, mask2, QWen_model, QWen_processor, device):
    """查询QWen进行mask选择"""
    # img64 = numpy_to_base64(image)
    # mask1_b64 = numpy_to_base64(mask1)
    # mask2_b64 = numpy_to_base64(mask2)  
    def encode_img(img):
        return numpy_to_base64(img)
    
    # 并行编码三个图像
    with ThreadPoolExecutor(max_workers=3) as executor:
        img64, mask1_b64, mask2_b64 = executor.map(encode_img, [image, mask1, mask2])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The image is this."},
                    {"type": "image", "image": f"data:image/png;base64,{img64}"},
                    {"type": "text", "text": "The MASK A is this."},
                    {"type": "image", "image": f"data:image/png;base64,{mask1_b64}"},
                    {"type": "text", "text": "The MASK B is this."},
                    {"type": "image", "image": f"data:image/png;base64,{mask2_b64}"},
                    {"type": "text", "text": f"""
                    CAMOUFLAGE MASK COMPARISON TASK
                    IMAGE: The image may contain a few animal/insect or human whose shape, color, texture, pattern and movement closely resemble its surroundings.
                    MASK A: Current best mask
                    MASK B: New candidate mask

                    MASK INTERPRETATION:
                    - White areas = potential camouflaged objects
                    - Black areas = background  

                    CRITICAL REQUIREMENT: The chosen mask MUST match your object analysis.

                    STEP-BY-STEP PROCESS:

                    1. OBJECT IDENTIFICATION:
                    - Carefully examine the image
                    - Identify all hidden/concealed objects and their exact locations

                    2. MASK EVALUATION:
                    Mask A: Does the white region cover all identified objects? How much extra background?
                    Mask B: Does the white region cover all identified objects? How much extra background?

                    3. SELECTION CRITERIA:
                    - PRIMARY: Choose the mask that covers ALL identified objects completely
                    - SECONDARY: Among masks that meet primary criterion, choose the one with least background
                    - If no mask covers all objects, choose the one that covers the most objects

                    4. CONSISTENCY CHECK:
                    - Ensure the chosen mask's white regions align with your identified objects
                    - If analysis says "2 camouflaged insects", the mask should have 2 corresponding white regions

                    OUTPUT JSON (DO NOT ADD ANY EXTRA INFO, JUST JSON!):
                    [{{
                        "better_mask": "Mask A" / "Mask B", 
                    }}]
                    """},
                ],
            } 
        ]
    
    text = QWen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = QWen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # 推理生成输出
    generated_ids = QWen_model.generate(**inputs, 
        max_new_tokens=64, 
        output_scores=True, 
        do_sample=True,
        temperature=0.5, 
        return_dict_in_generate=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
    ]
    
    output_text = QWen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print("QWen Output:", output_text)
    try:
        data = clean_and_parse_json(output_text)
        if isinstance(data, list):
            data = data[0]
        selected_mask = data["better_mask"]
        return selected_mask
    except:
        # print(f"解析失败，输出文本: {output_text}")
        # 默认返回Mask X
        return "Mask X"

def numpy_to_base64(img):
    """将numpy数组转换为base64字符串"""
    import cv2
    import base64
    import numpy as np
    
    # 处理数据类型转换
    if img.dtype == bool:
        # 将bool转换为uint8 (0和255)
        img = img.astype(np.uint8) * 255
    elif img.dtype == np.float32 or img.dtype == np.float64:
        # 将float转换为uint8 (0-255范围)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        # 其他类型转换为uint8
        img = img.astype(np.uint8)
    
    # 确保图像是2D或3D的
    if len(img.shape) == 2:
        # 灰度图转换为3通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # 单通道转换为3通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    success, buffer = cv2.imencode('.png', img)
    if success:
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    else:
        raise ValueError("图像编码失败")
    

def worker(rank, gpu_id, dataset, img_chunk, bbox_chunk, pred_dir, refine=True, merge=True, include_Qwen_pred=True):
    FOD_time = 0
    Seg_time = 0
    SMS_time = 0
    total_time = 0
    pbar = tqdm(
        total=len(img_chunk),
        desc=f"Worker {rank}, GPU: {gpu_id}",
        position=rank,  # 每个worker在不同的行显示
        leave=False  # 完成后不保留进度条
    )
    sam2_checkpoint = "/data/yilong/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "//data/yilong/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    QWen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/data/yilong/hf_Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        # device_map="auto"
        device_map={"": device},
    )
    QWen_processor = AutoProcessor.from_pretrained("/data/yilong/hf_Qwen2.5-VL-7B-Instruct")
    QWen_model.eval()
    print(f"Worker {rank} loading models on GPU {gpu_id}...")
    # 每个进程独立加载自己的模型
    dino_model = AutoModel.from_pretrained('/data/yilong/hf_dinov2', output_hidden_states=True)
    patch_size = dino_model.config.patch_size
    resolution = 0.5
    # 加载SAM模型
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    if "COD10K" in dataset:
        gt_dir = f"/data/yilong/Datasets/COD10K-v3/Test/GT_Object/"
    else:
        gt_dir = f"/data/yilong/Datasets/{dataset}/GT/"

    # 处理分配到的数据块
    for image_path, bboxes_qwen in zip(img_chunk,bbox_chunk):
        start_time = time.time()
        # setup_seed(43)
        # sam_pred_path = f'Preds/opt_prompt/{dataset}/{image_path.split("/")[-1].replace("jpg","png")}'
        sam_pred_path = f'/data/yilong/QWen_main/baseline/{dataset}/{image_path.split("/")[-1].replace("jpg","png")}'

        target_size=1022
        mask_save_path = os.path.join(pred_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        image, image_tensor, (pad_W, pad_H), (W,H) = load_and_pad_image(image_path, patch_size=patch_size, target_size=target_size)

        resolution = 0.5
        dino_model.eval()

        with torch.no_grad():
            outputs = dino_model(image_tensor)
            hidden_states = outputs.hidden_states
        # 提取 class token 和 patch tokens
        patch_tokens_2last = hidden_states[-1][:, 1:, :]
        _, _, pH, pW = image_tensor.shape
        num_patch_w = pW // patch_size  # 每行 patch 数
        num_patch_h = pH // patch_size  # 每列 patch 数
        feature_map_2last = patch_tokens_2last.reshape(1, num_patch_h, num_patch_w, -1).permute(0, 3, 1, 2)  # [1, D, H, W]
        candidate_masks, candidate_fgs, sim_maps_leiden, sim_maps_refine, padded_candidate_masks, low_res_mask, leiden_map, pca_data = get_patch_level_hierarchical_labels(
            data=feature_map_2last[0].cpu().numpy(),  # (768, H, W)
            pad_H=pad_H, pad_W=pad_W,
            n_pca = 16,
            resolution=resolution,
            n_neighbors=None,
            alpha=1,
            beta=6, gamma = 0.4
        )
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # cv2_img = np.array(image)

        if refine==True and merge==True:
            # print(f'refine==True and merge==True, {mask_prompt}')
            merged_sim_maps = merge_sim_maps(sim_maps_leiden+sim_maps_refine, thresh=0.95, visualisation=False)
        elif refine==True and merge==False:
            # merged_sim_maps = sim_maps_leiden+sim_maps_refine
            merged_sim_maps = sim_maps_refine+sim_maps_leiden
        elif refine==False and merge==False:
            # print(f'refine==False and merge==False, {mask_prompt}')
            merged_sim_maps = sim_maps_leiden
        elif refine==False and merge==True:
            # print(f'refine==False and merge==True, {mask_prompt}')
            merged_sim_maps = merge_sim_maps(sim_maps_leiden, thresh=0.95, visualisation=False)
            

        qwen_pred = cv2.imread(sam_pred_path,flags=0)>0
        orig_size = qwen_pred.shape
        qwen_pred = cv2.resize(qwen_pred.astype(np.uint8), dsize=(sim_maps_leiden[0].shape[1], 
                    sim_maps_leiden[0].shape[0]), 
                    interpolation=cv2.INTER_NEAREST)
        gt = cv2.imread(f'{gt_dir}{image_path.split("/")[-1].replace("jpg","png")}',flags=0)>0
        gt = cv2.resize(gt.astype(np.uint8), dsize=(sim_maps_leiden[0].shape[1], 
                    sim_maps_leiden[0].shape[0]), 
                    interpolation=cv2.INTER_NEAREST)

        if boundary_contact(qwen_pred, n=10)>0.75:
            qwen_pred = 1-qwen_pred
        qwen_pred = clean_mask(qwen_pred, min_size=100)

        end_time1 = time.time()
        FOD_time += (end_time1 - start_time)
        candidate_masks_final, candidate_scores_final = get_MAX_IoU_mask_from_SAM(merged_sim_maps, cv2_img, predictor, k=5)
        end_time2 = time.time()
        Seg_time += (end_time2 - end_time1)
        if include_Qwen_pred:
            candidate_masks_final.append(qwen_pred)
            candidate_scores_final.append(np.array(candidate_scores_final).mean())

        best_mask = pairwise_selection_from_QWen(np.array(image), candidate_masks_final, candidate_scores_final, QWen_model, QWen_processor, device=device)        
        final_pred = cv2.resize(best_mask.astype(np.uint8), dsize=(orig_size[1], orig_size[0]))
        end_time3 = time.time()
        SMS_time += (end_time3 - end_time2)
        save_mask_as_color(final_pred, mask_save_path)
        pbar.update(1)
        pbar.set_postfix({"current": image_path.split('/')[-1]})
    pbar.close()

    total_time += (end_time3 - start_time)

    print(f"Worker {rank} on GPU {gpu_id} finished in {total_time} seconds. FOD time: {FOD_time}, Seg time: {Seg_time}, SMS time: {SMS_time}")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="COD10K")
    parser.add_argument("--pred_dir", type=str, default="/data/yilong/QWen_main/Preds/")
    parser.add_argument("--json_file", type=str, default="/data/yilong/QWen_main/output/infer_CAMO_QWen.json")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--processes_per_gpu", type=int, default=1, help="Number of processes per GPU")
    parser.add_argument("--refine", action='store_true', help="enable refinement")
    parser.add_argument("--no_refine", dest='refine', action='store_false')

    parser.add_argument("--merge", action='store_true', help="enable merge")
    parser.add_argument("--no_merge", dest='merge', action='store_false')

    parser.add_argument("--include_qwen", action='store_true', help="enable qwen pred")
    parser.add_argument("--no_include_qwen", dest='include_qwen', action='store_false')

    parser.set_defaults(mask_prompt=True, refine=True, merge=True, include_qwen=False)

    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)

    # if args.dataset_name == "NC4K":
    #     img_dir = "/data/yilong/Datasets/NC4K/Imgs/"
    # elif args.dataset_name == "COD10K":
    #     img_dir = "/data/yilong/Datasets/COD10K-v3/Test/Image/"
    # elif args.dataset_name == "CAMO":
    #     img_dir = "/data/yilong/Datasets/CAMO/Images/Test/"
    # elif args.dataset_name == "CHAMELEON":
    #     img_dir = "/data/yilong/Datasets/CHAMELEON/Imgs/"

    pred_dir = args.pred_dir + args.dataset_name +f'/refine+{args.refine}_merge+{args.merge}_include+{args.include_qwen}'
    print(f'Prediction saved to {pred_dir}')
    os.makedirs(pred_dir, exist_ok=True)
    json_file = args.json_file

    img_list, bbox_list = get_img_list(json_file, pred_dir)
    # dinov2模型
    dino_model = AutoModel.from_pretrained('/data/yilong/hf_dinov2', output_hidden_states=True)
    image_processor = AutoImageProcessor.from_pretrained('/data/yilong/hf_dinov2')
    dino_model.eval()

    print("DINOv2 model loaded ...")
    patch_size = dino_model.config.patch_size
    resolution = 0.5



    # results = results[1063:]
    gpu_ids = [int(id) for id in args.gpus.split(',')]
    devices = [f'cuda:{id}' for id in gpu_ids]

    # Split work across processes
    num_gpus = len(gpu_ids)
    total_processes = num_gpus * args.processes_per_gpu
    chunk_size = len(img_list) // total_processes + 1
    
    # Create process pool
    pool = mp.Pool(processes=total_processes)
    # Distribute work
    processes = []
    for i in range(total_processes):
        gpu_id = gpu_ids[i % num_gpus]
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(img_list))
        chunk_imgs = img_list[start_idx:end_idx]
        chunk_bboxes = bbox_list[start_idx:end_idx]

        
        if not chunk_imgs:
            continue

        p = mp.Process(
            target=worker,
            args=(i, gpu_id, args.dataset_name, chunk_imgs, chunk_bboxes, pred_dir, args.refine, args.merge, args.include_qwen)
        )
        p.start()
        processes.append(p)
        # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("All processes completed successfully!")



    

            
