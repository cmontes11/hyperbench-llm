"""Merge a LoRA adapter into its base model for inference.

This script loads both the full-precision base model and its LoRA weights,
combines them into a single set of parameters, and saves the merged model
to disk. Use this after training to deploy without the PEFT dependency.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Base model HF repo or dir")
    parser.add_argument("--adapter", required=True, help="LoRA/PEFT adapter dir")
    parser.add_argument("--out", required=True, help="Output dir for merged model")
    args = parser.parse_args()
    
    # Load the base model, 16bit
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    
    print("Merging adapter weights into base model ...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.out} ...")
    merged_model.save_pretrained(args.out)
    # Save tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    tokenizer.save_pretrained(args.out)
    print("Done.")

if __name__ == "__main__":
    main()
