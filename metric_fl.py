from sklearn.metrics import classification_report, accuracy_score
import json
import os
output_results_file = os.path.join("saves/Qwen2.5-3B-Instruct/predict/train_steel_5", "eval_results.txt")

with open("saves/Qwen2.5-3B-Instruct/predict/train_steel_5/generated_predictions.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]


labels = [result["label"].strip() for result in results]
predictions = [result["predict"].strip() for result in results]


accuracy = accuracy_score(labels, predictions)
s = classification_report(labels, predictions, labels=list(set(labels)), digits=4)

with open(output_results_file, "w", encoding="utf-8") as f:
    f.write(str(s))
    f.write(f"Accuracy: {accuracy:.4f}\n")