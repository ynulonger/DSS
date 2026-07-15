import json
import re

input_path = "output/infer_COD10K_QWen_7B_results.json"
output_path = "output/infer_COD10K_QWen_7B_results_clean.json"

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

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "result" in item and isinstance(item["result"], str):
        item["result"] = clean_result_field(item["result"])

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已保存为规范JSON: {output_path}")