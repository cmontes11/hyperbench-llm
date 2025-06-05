"""Dual H100 NVL variant of ``train_scaling.py``.

This script mirrors ``train_scaling.py`` but tweaks the per-device batch size
for a two GPU setup. When launched with ``accelerate`` using two processes it
keeps the effective global batch size consistent with the single-GPU version.

Example accelerate config (``accelerate config``):

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
mixed_precision: bf16

Launch with:

    accelerate launch scripts/train_scaling_h100nvl.py --rows <NUM_ROWS>
"""

import argparse
import json
import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ----- CLI -----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--rows",
    type=int,
    default=10_000,
    help="number of training rows to use",
)
args = parser.parse_args()

# ----- 4-bit quantized model -----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)

hf_token = os.getenv("HF_TOK")
if hf_token is None:
    raise EnvironmentError(
        "HF_TOK environment variable not set."
    )

tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_auth_token=hf_token,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_cfg,
    device_map="auto",
    use_auth_token=hf_token,
)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ----- LoRA adapter -----
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# ----- dataset -----
ds = load_dataset("bigcode/the-stack-smol", split=f"train[:{args.rows}]")
# A small held-out slice for evaluation
eval_ds = load_dataset("bigcode/the-stack-smol", split="train[100000:101000]")

def tok_func(batch):
    return tok(
        batch["content"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = ds.map(tok_func, batched=True, remove_columns=ds.column_names)
tokenized_eval = eval_ds.map(tok_func, batched=True, remove_columns=eval_ds.column_names)

# ----- training args -----
training_args = TrainingArguments(
    output_dir="outputs/scaling_run_h100nvl",
    per_device_train_batch_size=64,  # 16 * 2 GPUs = 32 global
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=3e-4,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=50,
    evaluation_strategy="epoch",
    include_tokens_per_second=True,
    save_strategy="epoch",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
)


def collator(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["input_ids"] for b in batch]),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized_eval,
    data_collator=collator,
)

os.makedirs("outputs/scaling_run_h100nvl", exist_ok=True)
train_result = trainer.train()
trainer.save_model("outputs/scaling_run_h100nvl/lora_adapter")

eval_metrics = trainer.evaluate()
seq_len = len(tokenized[0]["input_ids"])
tokens_seen = seq_len * len(tokenized) * training_args.num_train_epochs
metrics = {**train_result.metrics, **eval_metrics,
           "tokens_total": tokens_seen, "tokens_seen": tokens_seen}
with open("outputs/scaling_run_h100nvl/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Training finished, metrics written to outputs/scaling_run_h100nvl")
