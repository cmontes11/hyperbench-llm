from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer  
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training   
from datasets import load_dataset            
import torch                                      
import os

# load 4 bit
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)

hf_token = os.getenv("HF_TOK")
if hf_token is None:
    raise EnvironmentError(
        "HF_TOK environment variable not set. Please set it to your HuggingFace token "
        "before running this script."
    )

tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_auth_token=hf_token
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    use_auth_token=hf_token
)

print("GPU footprint after load:",
      base_model.get_memory_footprint() / 1e9, "GB")
      
# fix pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    base_model.config.pad_token_id = tok.eos_token_id
    
base_model.config.use_cache = False  
base_model = prepare_model_for_kbit_training(base_model)

# add lora adapter
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
model.print_trainable_parameters()        # check trainable params
model.gradient_checkpointing_enable()     # checkpoint long training

# example code dataset, small for example
ds = load_dataset("bigcode/the-stack-smol", split="train[:10000]")

def tok_func(examples):
    return tok(
        examples["content"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = ds.map(tok_func, batched=True, remove_columns=ds.column_names)

# training setup
args = TrainingArguments(
    output_dir="outputs/naive_run",
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=4,         
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,                        
    optim="paged_adamw_8bit",
    logging_steps=25,
    include_tokens_per_second=True,
    save_strategy="epoch"
)

def data_collator(batch):
    return {
        "input_ids":  torch.tensor([x["input_ids"]  for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["input_ids"]  for x in batch])
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("outputs/naive_run/lora_adapter")

print("training finished, adapter saved to outputs/naive_run")

