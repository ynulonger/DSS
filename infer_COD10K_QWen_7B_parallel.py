import re
import os
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Process
import torch.multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if "content" in message:
            for content_item in message["content"]:
                if content_item.get("type") == "image":
                    img_path = content_item["image"]
                    image_inputs.append(Image.open(img_path).convert("RGB"))
                elif content_item.get("type") == "video":
                    video_inputs.append(content_item["video"])
    return image_inputs, video_inputs

def infer_worker(rank, gpu_id, image_paths, dataset_name, output_dir, model_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id*2)+','+str(gpu_id*2+1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    results = []
    start = time.time()
    for img_path in tqdm(image_paths, desc=f"[Rank {rank}, GPU {gpu_id}] Processing", position=rank, leave=True):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": """The image may contain a few animal/insect or human whose shape, color, texture, pattern and movement 
                            closely resemble its surroundings. Please identify them and provide their locations in the format of coordinates, as precisely 
                            as possible. Also provide me with the confidence of your localisation result. The output should be in JSON format, eg: 
                            "{"
                               "bbox_2d": [[x1, y1, x2, y2],[x1, y1, x2, y2]],
                               "label": "dog",
                               "confidence": "confidence value in between [0,1]"
                            "}". 
                            If you can not locate the camouflaged/concealed/hidden objects, the format of output should be in JSON format, eg:
                            "{"
                                "description": "suggest possible camouflaged/concealed/hidden ainmals/insects/human"
                            }. DO NOT ADD ANY EXTRA INFO, JUST JSON!"""}
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        with torch.no_grad():
            try:
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            except:
                print(f"Error: {img_path}")
                continue
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results.append({
            "image": img_path,
            "result": output_text[0] if output_text else ""
        })
        torch.cuda.empty_cache()
        # print(f"[GPU {gpu_id}] Processed: {img_path}")
    # 保存本进程结果
    out_path = os.path.join(output_dir, f"infer_{dataset_name}_QWen_7B_{rank}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    end = time.time()
    print(f"[GPU {gpu_id}] Inference time: {end - start:.2f} seconds")
    print(f"[GPU {gpu_id}] Results saved to {out_path}")

def clean_result_field(result_str):
        # 去除 markdown 代码块标记和多余空白
        result_str = result_str.strip()
        # 去除 ```json ... ``` 或 ``` ... ```
        result_str = re.sub(r"^```json|^```|```$", "", result_str, flags=re.MULTILINE).strip()
        # 尝试解析为json
        try:
            return json.loads(result_str)
        except Exception:
            return result_str  # 如果解析失败，保留原字符串

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

if __name__ == "__main__":
    seed = 42  # 选择一个固定值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    os.environ['PYTHONHASHSEED'] = str(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/data/yilong/Datasets/COD10K-v3/Test/Image/")
    parser.add_argument("--model_dir", type=str, default="/data/yilong/hf_Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/data/yilong/QWen_main/output")
    parser.add_argument("--dataset", type=str, default="COD10K")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    mp.set_start_method('spawn', force=True)

    args = parser.parse_args()

    gpus = args.gpus.split(',')

    image_list = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_list.sort()
    if "COD10K" in args.dataset:
        image_list = image_list[:2026]
    # image_list = image_list[:40]

    chunks = split_list(image_list, len(gpus))
    # chunks = split_list(chunks[2], 8)
    # chunks = split_list(chunks[6], 8)

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(gpus, chunks)):
        p = Process(target=infer_worker, args=(rank, gpu_id, chunk, args.dataset, args.output_dir, args.model_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # '''
    # 合并所有json
    all_results = []
    for rank in range(len(gpus)):
        part_file = os.path.join(args.output_dir, f"infer_{args.dataset}_QWen_7B_{rank}.json")
        with open(part_file, "r", encoding="utf-8") as f:
            all_results.extend(json.load(f))
            os.remove(part_file)

    merged_path = os.path.join(args.output_dir, f"infer_{args.dataset}_QWen_7B.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"All results merged to {merged_path}")

    # 清理结果字段
    input_path = merged_path
    output_path = os.path.join(args.output_dir, f"infer_{args.dataset}_QWen_7B_clean.json")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "result" in item and isinstance(item["result"], str):
            item["result"] = clean_result_field(item["result"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.remove(input_path)
    print(f"已保存为规范JSON: {output_path}") 
    # '''
