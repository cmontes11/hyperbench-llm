"""Fine-tune Llama-3 8B with optimized settings for A100 GPUs.

The script demonstrates a fast LoRA training setup using 4-bit quantization
and a high batch size. It is intended for quick experimentation on a single
GPU and saves only the adapter weights after training.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

# Load in 4-bit with double quantization for memory efficiency
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)

# Read Huggingface token for model hub
hf_token = os.getenv("HF_TOK")

# Load tokenizer
tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_auth_token=hf_token
)

# Load quantized base model on appropriate device
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    use_auth_token=hf_token
)

print("GPU footprint after load:",
      base_model.get_memory_footprint() / 1e9, "GB")

# Handle missing pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    base_model.config.pad_token_id = tok.eos_token_id

# Disable KV cache 
base_model.config.use_cache = False

# Prepare for parameter-efficient training
base_model = prepare_model_for_kbit_training(base_model)

# Add LoRA adapters 
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()        # Print trainable params for inspection

# Load tiny example code dataset (replace with your real data as needed)
ds = load_dataset("bigcode/the-stack-smol", split="train[:10000]")

def tok_func(examples):
    return tok(
        examples["content"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = ds.map(tok_func, batched=True, remove_columns=ds.column_names)

# Speed-tuned TrainingArguments for A100
args = TrainingArguments(
    output_dir="outputs/naive_run",
    per_device_train_batch_size=32,          # High mbs with high ram
    gradient_accumulation_steps=1,           # Lower steps, since batch size is big
    num_train_epochs=1,                   
    learning_rate=2e-4,
    bf16=True,                               # Use bfloat16 (A100-native type)
    optim="paged_adamw_8bit",                # Highly efficient optimizer for 4/8-bit training
    logging_steps=50,                        # Reduce logging overhead 
    include_tokens_per_second=True,          # Log throughput stats
    save_strategy="epoch",                   # Save only at epoch end
    dataloader_num_workers=8,                # Increases data loading speed
    dataloader_pin_memory=True,              # Speeds up host->GPU transfer                            
)

def data_collator(batch):
    return {
        "input_ids":        torch.tensor([x["input_ids"]  for x in batch]),
        "attention_mask":   torch.tensor([x["attention_mask"] for x in batch]),
        "labels":           torch.tensor([x["input_ids"]  for x in batch])
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# Train the model at max speed
trainer.train()

# Save just the LoRA adapter weights after training
trainer.save_model("outputs/naive_run/lora_adapter")

print("Training finished, adapter saved to outputs/naive_run")
