# hyperbench-llm

Tools and utilities for benchmarking and optimizing [**Llama‑8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) so it can run efficiently on a single RTX&nbsp;4080.

The project aims to understand the memory and performance trade‑offs when deploying Llama‑8B on consumer GPUs. Training or fine‑tuning typically happens on powerful cloud hardware, while inference and benchmarking take place locally. The scripts below measure memory usage, merge LoRA adapters, and provide simple training helpers so you can experiment with different setups.


## Version Highlights
- **v0.3 – Compression Experiments**
  Quantization cut model size from 8 GB to 4.8 GB and boosted throughput from 17.5 → 108 tok/s at 3.27 perplexity.
- **v0.2 – Scaling Laws**
  Trained 1k–40k row datasets on dual H100 NVL GPUs for ~2h ($10), locating diminishing returns after 20k rows.
- **v0.1 – Training Helpers**
  Swept 22 parameter configs in ~5h with two W&B agents (A100 + H100) for about $25.
- **v0.0 – Baseline Utilities**
  Baseline Llama‑8B: 17.5 tok/s and 3.69 perplexity on a single RTX 4080.
---

## Files and Descriptions

Scripts are in **`scripts/`**, configurations in **`configs/`**, and figures are in **`images/`**.

* **scripts/bench.py** – Benchmark a model to measure speed, VRAM use, parameter counts and perplexity on a dataset split.
* **scripts/gpu_memory_test.py** – Show detailed GPU memory footprint for a given model.
* **scripts/merge_lora.py** – Merge a LoRA/PEFT adapter into a base model for faster inference.
* **scripts/train_fast.py** – Speed-tuned training for A100 GPUs.
* **scripts/train_h100.py** – Sweep-ready training script for H100 GPUs.
* **scripts/train_naive.py** – Minimal example for finetuning or adapter training.
* **scripts/train_scaling.py** – Variable dataset training for scaling-law experiments using the sweep's best hyper-parameters.
* **scripts/train_scaling_h100nvl.py** – Dual-GPU variant for H100 NVL systems.
* **configs/qlora_loss.yaml** – Example W&B sweep configuration.
* **environment.yml** – Conda environment specification with all required packages.
* **images/sweep results.png** – Example sweep results image.
* **images/scaling analysis.png** – Example scaling analysis chart.
* **images/compression analysis.png** – Four-panel compression analysis chart.
* **scripts/eval_slicer.py** – Save a slice of the evaluation dataset.

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
   python scripts/bench.py --ckpt <MODEL_DIR_OR_HF_NAME> --ppl_dataset mbpp --ppl_split train[:1024] --out result.json
   ```
2. Merge LoRA adapter weights:
   ```bash
   python scripts/merge_lora.py --ckpt <BASE_MODEL> --adapter <ADAPTER_DIR> --out <OUTPUT_DIR>
   ```
3. Check GPU memory usage:
   ```bash
   python scripts/gpu_memory_test.py
   ```
4. Train or fine-tune (optional):
   ```bash
   accelerate launch scripts/train_naive.py
   ```
   Training metrics are logged to [Weights & Biases](https://wandb.ai/) if the `wandb` package is installed and configured.
5. Run the optimized A100 script:
   ```bash
   accelerate launch scripts/train_fast.py
   ```
6. Run the variable dataset script (uses the sweep's best hyper-parameters):
   ```bash
   accelerate launch scripts/train_scaling.py --rows 50000
   ```
7. Run the H100 parameter sweep with Weights & Biases:
   ```bash
   wandb sweep configs/qlora_loss.yaml
   wandb agent <ENTITY/PROJECT/SWEEP_ID>
   ```
   The agent executes `python scripts/train_h100.py` with its chosen arguments.

Before running GPU tests or training, set the `HF_TOK` environment variable with your Hugging Face token so the scripts can download the model.

## Dataset Recommendations

For fast code benchmarking consider the following datasets:

- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)
- [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp)

## Version 0.0 – Baseline Benchmarks

The following results were obtained on a single RTX&nbsp;4080 using
`bigcode/the-stack-smol` to evaluate perplexity. The fine‑tuned model
was produced by merging its LoRA adapter into the base weights.

### Base Model

```bash
python scripts/bench.py --ckpt meta-llama/Meta-Llama-3-8B-Instruct \
  --ppl_dataset bigcode/the-stack-smol \
  --ppl_split train[:1024] \
  --out base_bench.json
```

Output:

```json
{
  "ram_footprint_gb": 5.59,
  "gpu_peak_gb": 7.24,
  "tok_per_sec": 17.75,
  "latency_ms_first_tok": 56.3,
  "elapsed_s": 14.42,
  "perplexity": 3.69
}
```

### Merged Fine‑tuned Model

```bash
python scripts/bench.py --ckpt merged_model_dir \
  --ppl_dataset bigcode/the-stack-smol \
  --ppl_split train[:1024] \
  --out merged_bench.json
```

Output:

```json
{
  "ram_footprint_gb": 5.59,
  "gpu_peak_gb": 7.24,
  "tok_per_sec": 18.08,
  "latency_ms_first_tok": 55.3,
  "elapsed_s": 14.16,
  "perplexity": 2.42
}
```

The merged model improves perplexity from **3.69** to **2.42** while
keeping memory usage constant and slightly increasing tokens per second.

## Version 0.1 – Parameter Sweep & Training Helpers

This release introduces an automated hyper-parameter sweep using
[Weights & Biases](https://wandb.ai/). The sweep configuration lives in
`configs/qlora_loss.yaml` and trains with `scripts/train_h100.py`.

Run the sweep:

```bash
wandb sweep configs/qlora_loss.yaml
wandb agent <ENTITY/PROJECT/SWEEP_ID>
```

Each agent invocation runs `python scripts/train_h100.py` with the selected
arguments and logs metrics to W&B.

The best run achieved the lowest training loss with:

```
lora_alpha lora_r     lr   train_loss
       128      16 0.0003      0.7806
```

![Sweep results](images/sweep%20results.png)

This figure shows mean loss per each hyperparameter: LoRA Alpha, LoRA Rank, Learning Rate. The top and middle plots both show minimal change across their respective hyperparameter however the bottom plot indicates strongly that as the learning rate decreases the mean loss also decreases as well as the highest and lowest values at that parameter value. This indicates that a lower learning rate is linked to increased model performance for this configuration.

## Version 0.2 – Scaling Law Experiments
Files used: `scripts/train_scaling_h100nvl.py`, `scripts/train_scaling.py`, `scripts/merge_lora.py`, and `scripts/bench.py`.

The plot below was produced by training adapters on dataset sizes from 1k–40k rows. After each run the adapters were merged and benchmarked:
```bash
# Train
accelerate launch scripts/train_scaling_h100nvl.py --rows 40000
# Merge
python scripts/merge_lora.py --ckpt meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter outputs/scaling_run_h100nvl/lora_adapter_40k \
  --out outputs/merged_models/merged_40k
# Bench
python scripts/bench.py --ckpt outputs/merged_models/merged_40k \
  --ppl_dataset bigcode/the-stack-smol \
  --ppl_split train[100000:101000] \
  --out outputs/merged_models/merged_40k.json
```

![Scaling analysis](images/scaling%20analysis.png)

This figure shows four scaling curves on log–log axes. Training with just
1k rows appears as the smallest point, while 40k rows is the largest. The
lines follow a mostly linear trend, illustrating the scaling law. However,
the final curve flattens, meaning the model stops improving after 20k rows
and gains little from 40k.
## Version 0.3 – Compression & Quantization
Files used: `scripts/merge_lora.py`, `scripts/bench.py` and the `llama.cpp` container.

To measure the quantization trade-offs:
```bash
# Quantize the model (e.g. to Q6_K)
./bin/llama-quantize <Uncompressed Model Location> <Quantized Model Location> Q6_K

# Perplexity test
./bin/llama-perplexity -m <Quantized Model Location> -f eval_slice.txt -n 128 -t 8

# Tokens/sec benchmark
./bin/llama-bench -m <Quantized Model Location> -n 512
```

![Compression analysis](images/compression%20analysis.png)

This figure contains four subplots:
1. **Top left** : tokens per second vs. perplexity showing the speed/accuracy trade-off. The best configuration is marked with a star. Note: exponential increase in perplexity vs tokens/sec.
2. **Top right** : tokens per second vs. model size in gigabytes with the best model starred. Note: linear relationship between model size and tokens/sec.
3. **Bottom left** : inference speed by model where each point is labeled with its size in gigabytes and includes error bars for variability.
4. **Bottom right** : perplexity for each model, again with the best one highlighted. Note: exponential increase in perplexity vs model size.



## Completed

### v0.0
- [x] Benchmarking utilities
- [x] LoRA adapter merging
- [x] GPU memory profiler

### v0.1
- [x] Naive training example
- [x] Environment setup guide
- [x] Parameter sweep utilities

### v0.2
- [x] Scaling law experiments


### v0.3
- [x] Deploy to RTX 4080 for maximizing tokens/sec
- [x] Quantization experiments
- [x] Model compression techniques


## TODO

### v0.4
- [ ] Custom datasets

### v0.5
- [ ] Custom models

### v0.6
- [ ] Distillation training


