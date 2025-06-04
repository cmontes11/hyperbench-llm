"""Variable dataset LoRA training example for scaling experiments.

This script extends the naive/fast examples to allow a custom dataset
size via the ``--rows`` argument. It keeps the 4-bit quantized loading
and LoRA setup, while using fast training defaults suitable for a
single powerful GPU (e.g. H100). The LoRA hyper-parameters and learning
rate use the best values found in the README sweep (`r=16`,
`lora_alpha=128`, `lr=3e-4`). Adjust ``batch_size`` and other
parameters as needed for your hardware. For maximum throughput on an
H100 consider enabling ``bf16`` and fused optimizers.
"""

import argparse
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
parser.add_argument("--rows", type=int, default=10_000,
                    help="number of training rows to use")
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

def tok_func(batch):
    return tok(
        batch["content"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = ds.map(tok_func, batched=True, remove_columns=ds.column_names)

# ----- training args -----
training_args = TrainingArguments(
    output_dir="outputs/scaling_run",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=3e-4,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=50,
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
    data_collator=collator,
)

trainer.train()
trainer.save_model("outputs/scaling_run/lora_adapter")

print("Training finished, adapter saved to outputs/scaling_run")
