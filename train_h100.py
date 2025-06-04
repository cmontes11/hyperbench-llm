#!/usr/bin/env python
"""
Param-sweep-ready LoRA fine-tune for Llama-3 8B on H100 NVL.

Every tunable hyper-parameter is exposed on the CLI so that
WandB’s sweep agent can drive the grid / Bayesian search.
"""

import argparse, os, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--dataset_path", type=str, default="bigcode/the-stack-smol")
p.add_argument("--rows",        type=int, default=10_000)
p.add_argument("--max_seq_len", type=int, default=512)
p.add_argument("--batch_size",  type=int, default=64)
p.add_argument("--epochs",      type=int, default=1)
p.add_argument("--lora_r",      type=int, default=16)
p.add_argument("--lora_alpha",  type=int, default=32)
p.add_argument("--lr",          type=float, default=1e-4)
p.add_argument("--output",      type=str,  default="outputs/sweep_run")
p.add_argument("--bf16",        type=lambda x: (str(x).lower() == 'true'), default=False)
p.add_argument("--fp8",         type=lambda x: (str(x).lower() == 'true'), default=False)
args = p.parse_args()

# ---------- 4-bit load ----------
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)

tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_auth_token=os.getenv("HF_TOK")
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    use_auth_token=os.getenv("HF_TOK")
)

# pad-token fix
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

# ---------- LoRA ----------
lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj",
        "o_proj","gate_proj","up_proj","down_proj"
    ],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# ---------- dataset ----------
ds = load_dataset(args.dataset_path, split=f"train[:{args.rows}]")

def tok_func(examples):
    return tok(
        examples["content"],
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length"
    )

tokenized = ds.map(tok_func, batched=True, remove_columns=ds.column_names)

# ---------- training args ----------
training_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    bf16=args.bf16,
    fp16=False,
    optim="adamw_torch_fused",
    logging_steps=25,
    include_tokens_per_second=True,
    save_strategy="epoch",
    dataloader_num_workers=8,
    dataloader_pin_memory=True
)

def collator(batch):
    return {
        "input_ids":      torch.tensor([b["input_ids"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
        "labels":         torch.tensor([b["input_ids"] for b in batch])
    }

Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator
).train()
