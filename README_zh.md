## 运行环境与核心依赖

为确保实验结果的科学可复现性，所有涉及大语言模型分类性能的实验均在以下基准软硬件环境中执行：

* **运行环境与核心语言** Python 3.10.18
* **深度学习核心框架** PyTorch 2.7.1
* **大模型生态依赖库** Transformers 4.52.4
* **LLaMA-Factory**  0.9.5.dev0

### 编译安装步骤

本框架依托 LLaMA-Factory 进行底层分布式训练支持。请严格执行以下流水线进行环境拓扑与基准度量指标库的编译：
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt

```

## 数据集预处理与配置注册

本研究贡献的工业基准数据集包含 1,070 条由钢铁行业专家深度标注的高质量样本，覆盖七个细粒度工业维度。

### 1. 五折交叉验证切分

执行数据管线中的预处理脚本，将原始样本切分为五折分布，并自动注入结构化的指令提示：

```bash
python steel_dataset/process.py

```
预处理完成后将生成 `train_steel_i` 与 `test_steel_i`。

### 2. 学术注意事项

为使 C-AUM 优化器能够精确捕获逐样本的动态学习行为，每个样本数据流中必须包含且仅包含一个唯一的 `"sample_ids"` 键值对。该全局唯一标识符用于跨 epoch 连续追踪难样本的语义边界。

### 3. 元数据注册表配置

在正式启动训练前，须在 `"C-AUM/data/dataset_info.json"` 文件中显式配置切分后的数据路径，以确保数据读取引擎准确挂载。


## 训练管线与推理评估

在微调与推理阶段，本研究在各底层模型间分别保持了完全对齐的参数设置，以确保经验对比的绝对公平。

### 1. 训练阶段

```bash
llamafactory-cli train configs/llama3_lora_sft.yaml

```

### 2. 推理阶段

```bash
llamafactory-cli train configs/llama3_lora_predict.yaml

```

### 3. 量化指标评估

```bash
python metric_fl.py

```
