# hyperbench-llm

**Version 0.1**
Tools and utilities for benchmarking and optimizing [**Llama‑8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) so it can run efficiently on a single RTX&nbsp;4080.

The project aims to understand the memory and performance trade‑offs when deploying Llama‑8B on consumer GPUs. Training or fine‑tuning typically happens on powerful cloud hardware, while inference and benchmarking take place locally. The scripts below measure memory usage, merge LoRA adapters, and provide simple training helpers so you can experiment with different setups.

---

## Files and Descriptions

* **bench.py** – Benchmark a model to measure speed, VRAM use, parameter counts and perplexity on a dataset split.
* **merge_lora.py** – Merge a LoRA/PEFT adapter into a base model for faster inference.
* **gpu_memory_test.py** – Show detailed GPU memory footprint for a given model.
* **train_naive.py** – Minimal example for finetuning or adapter training.
* **train_fast.py** – Speed-tuned training for A100 GPUs.
* **train_h100.py** – Sweep-ready training script for H100 GPUs.
* **qlora_loss.yaml** – Example W&B sweep configuration.
* **environment.yml** – Conda environment specification with all required packages.
* **sweep results.png** – Example sweep results image.

---

## Environment Setup

1. Install [Conda](https://docs.conda.io/).
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate it:
   ```bash
   conda activate qlora
   ```

## Quickstart

1. Benchmark a base or merged model:
   ```bash
   python bench.py --ckpt <MODEL_DIR_OR_HF_NAME> --ppl_dataset mbpp --ppl_split train[:1024] --out result.json
   ```
2. Merge LoRA adapter weights:
   ```bash
   python merge_lora.py --ckpt <BASE_MODEL> --adapter <ADAPTER_DIR> --out <OUTPUT_DIR>
   ```
3. Check GPU memory usage:
   ```bash
   python gpu_memory_test.py
   ```
4. Train or fine-tune (optional):
   ```bash
   accelerate launch train_naive.py
   ```
   Training metrics are logged to [Weights & Biases](https://wandb.ai/) if the `wandb` package is installed and configured.
5. Run the optimized A100 script:
   ```bash
   accelerate launch train_fast.py
   ```
6. Run the H100 parameter sweep with Weights & Biases:
   ```bash
   wandb sweep qlora_loss.yaml
   wandb agent <ENTITY/PROJECT/SWEEP_ID>
   ```
   The agent executes `python train_h100.py` with its chosen arguments.

Before running GPU tests or training, set the `HF_TOK` environment variable with your Hugging Face token so the scripts can download the model.

## Dataset Recommendations

For fast code benchmarking consider the following datasets:

- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)
- [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp)

## Parameter Sweep Results

This release introduces an automated hyper-parameter sweep using
[Weights & Biases](https://wandb.ai/). The sweep configuration lives in
`qlora_loss.yaml` and trains with `train_h100.py`.

Run the sweep:

```bash
wandb sweep qlora_loss.yaml
wandb agent <ENTITY/PROJECT/SWEEP_ID>
```

Each agent invocation runs `python train_h100.py` with the selected
arguments and logs metrics to W&B.

The best run achieved the lowest training loss with:

```
lora_alpha lora_r     lr   train_loss
       128      16 0.0003      0.7806
```

![Sweep results](sweep%20results.png)

## Completed

- [x] Benchmarking utilities
- [x] LoRA adapter merging
- [x] GPU memory profiler
- [x] Naive training example
- [x] Environment setup guide
- [x] Parameter sweep utilities

## TODO

- [ ] Scaling law experiments
- [ ] Model compression techniques


