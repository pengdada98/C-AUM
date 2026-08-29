import pandas as pd
import json
import random
import os

# It is recommended to use an absolute path
excel_path = "C-AUM/steel_dataset/steel.xlsx" 
output_root = "C-AUM/steel_dataset"

if not os.path.exists(output_root):
    os.makedirs(output_root)

seed = 42
k_folds = 5  

# prompt
instruct_template = '''你是一名资深的文本分类专家，请对给定的领域文本进行分类，文本类型必须从给定选项中选择。
给定选项：牌号、产地、品种、标准、缺陷、用途、性能
领域文本：'''


valid_labels = {'牌号', '产地', '品种', '标准', '缺陷', '用途', '性能'}


random.seed(seed)


df = pd.read_excel(excel_path, header=None)


texts = df.iloc[:, 0].astype(str).tolist()
labels = df.iloc[:, 1].astype(str).tolist()

selected_data = []
for text, label in zip(texts, labels):
    label = label.strip() # 去除可能存在的首尾空格
    if label in valid_labels:
        selected_data.append((text, label))
    else:
        print(f"Warning: Unknown label category '{label}' detected. This data instance has been skipped.")


random.shuffle(selected_data)


def convert_to_json(dataset):
    result = []
    for idx, (text, label) in enumerate(dataset):
        result.append({
            "sample_ids": idx,
            "instruction": instruct_template + text,
            "input": "",
            "output": label
        })
    return result

total_len = len(selected_data)
fold_size = total_len // k_folds


for i in range(k_folds):
    start = i * fold_size
    
    end = (i + 1) * fold_size if i != k_folds - 1 else total_len
    

    test_list = selected_data[start:end]
    train_list = selected_data[:start] + selected_data[end:]
    

    train_json = convert_to_json(train_list)
    test_json = convert_to_json(test_list)
    
    train_path = os.path.join(output_root, f"train_steel_{i+1}.json")
    test_path = os.path.join(output_root, f"test_steel_{i+1}.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)
        
    print(f"Fold {i+1} completed: {len(train_json)} training samples, {len(test_json)} test samples.")

print(f"\nAll data has been successfully saved to the directory: {output_root}")