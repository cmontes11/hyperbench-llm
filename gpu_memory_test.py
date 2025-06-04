from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

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

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    use_auth_token=hf_token  
)

print("GPU footprint:", model.get_memory_footprint()/1e9, "GB")
