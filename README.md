# hyperbench-llm
repo for benchmarking and optimizing llama 8B for running on a RTX 4080

Tools and scripts for optimizing, benchmarking, and running large language models (LLMs) for desktop GPUs, with support for adapter (LoRA/PEFT) merging and memory analysis.  
Designed for workflows where you train/tune on powerful cloud GPUs (A100s) and run/infer on consumer hardware (e.g., RTX 4080).

---

## Files and Descriptions

| Filename         | Description                                                                          |
|------------------|--------------------------------------------------------------------------------------|
| `bench.py`       | Benchmark script for LLMs. Measures speed, VRAM use, parameter counts, and perplexity on a chosen dataset split. Supports evaluating base, merged, or adapter models.         |
| `merge_lora.py` | Merges a LoRA/PEFT adapter into a base model, producing a self-contained, fast model for deployment and easy benchmarking.                           |
| `gpu_memory_test.py` | Reports detailed GPU memory usage for a given model (can help when tuning for your GPU).                                           |
| `train_naive.py`       | Example or entry-point script for finetuning/training an LLM adapter on a specific dataset or set of hyperparameters.           |
| `environment.yml`| Conda environment file listing all dependencies required to run the above scripts. Import with `conda env create -f environment.yml`.     |

---

## Quickstart

**Set up the environment**  
   ```bash
   >>conda env create -f environment.yml
   >>conda activate [your-env-name]

1.	Benchmark a base or merged model
>>python bench.py --ckpt [MODEL_DIR or HF NAME] --ppl_dataset mbpp --ppl_split train[:1024] --out result.json
2.	Merge LoRA adapter weights for deployment/inference
>>python merge_lora.py 
3.	Check memory usage
>>python memory_test.py 
4.	Train/fine-tune (if applicable)
>>accelerate launch train_naive.py

Dataset Recommendations
•	For fast code benchmarking: bigcode/the-stack-smol (subset), mbpp
