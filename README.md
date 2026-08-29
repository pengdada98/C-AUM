## Operating Environment and Core Dependencies

To ensure the scientific reproducibility of the experimental results, all experiments involving the classification performance of large language models are executed in the following baseline software and hardware environment:

* **Operating Environment and Core Language** Python 3.10.18
* **Deep Learning Core Framework** PyTorch 2.7.1
* **LLM Ecosystem Dependency Library** Transformers 4.52.4
* **LLaMA-Factory**  0.9.5.dev0

### Compilation and Installation Steps

This framework relies on LLaMA-Factory for underlying distributed training support. Please strictly execute the following pipeline to compile the environment topology and baseline metrics library:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt

```

## Dataset Preprocessing and Configuration Registration

The industrial benchmark dataset contributed by this study contains 1,070 high-quality samples deeply annotated by steel industry experts, covering seven fine-grained industrial dimensions.

### 1. Five-Fold Cross-Validation Split

Execute the preprocessing script in the data pipeline to split the raw samples into a five-fold distribution and automatically inject structured instruction prompts:

```bash
python steel_dataset/process.py

```

Upon completion of the preprocessing, `train_steel_i` and `test_steel_i` will be generated.

### 2. Academic Notes

To enable the C-AUM optimizer to accurately capture the dynamic learning behavior of individual samples, each sample data stream must contain one and only one unique `"sample_ids"` key-value pair. This globally unique identifier is used to continuously track the semantic boundaries of hard examples across epochs.

### 3. Metadata Registry Configuration

Before officially launching the training, the split data paths must be explicitly configured in the `"C-AUM/data/dataset_info.json"` file to ensure the accurate mounting of the data reading engine.

## Training Pipeline and Inference Evaluation

During the fine-tuning and inference phases, this study maintains completely aligned parameter settings across the underlying models to ensure the absolute fairness of the empirical comparison.

### 1. Training Phase

```bash
llamafactory-cli train configs/llama3_lora_sft.yaml

```

### 2. Inference Phase

```bash
llamafactory-cli train configs/llama3_lora_predict.yaml

```

### 3. Quantitative Metric Evaluation

```bash
python metric_fl.py

```