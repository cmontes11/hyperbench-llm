import time
import torch
import json
import argparse
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

def load_model(checkpoint, adapter=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"
    )
    tok = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, 
        quantization_config=bnb, 
        device_map="auto"
    )
    if adapter and PeftModel:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok

def memory_metrics(model):
    torch.cuda.empty_cache()
    alloc = torch.cuda.max_memory_allocated()
    return {
        "ram_footprint_gb": round(getattr(model, "get_memory_footprint", lambda: 0)() / 1e9, 2),
        "gpu_peak_gb": round(alloc / 1e9, 2),
    }

def latency_metrics(model, tokenizer, prompt="Hello", max_new=256):
    model.eval()
    with torch.inference_mode():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new)
        torch.cuda.synchronize()
        delta = time.perf_counter() - t0
        n_generated = out.shape[-1] - ids.shape[-1]
        return {
            "tok_per_sec": round(n_generated / delta, 2) if n_generated > 0 else 0,
            "latency_ms_first_tok": round(delta * 1000 / n_generated, 1) if n_generated > 0 else 0,
            "elapsed_s": round(delta, 2),
        }

def _pick_text_column(ds):
    for col in ds.column_names:
        if "text" in col and isinstance(ds[col][0], str):
            return col
        if "content" in col and isinstance(ds[col][0], str):
            return col
    for col in ds.column_names:
        if isinstance(ds[col][0], str):
            return col
    raise RuntimeError(f"No text or string column found in dataset {ds.column_names}")

def compute_perplexity(model, tokenizer, dataset_name, split, batch_size=8, max_length=512):
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset(dataset_name, split=split)
    texts = ds[_pick_text_column(ds)]
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(n_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            encodings = tokenizer(
                batch_texts, 
                return_tensors='pt',
                padding='max_length', 
                truncation=True, 
                max_length=max_length
            )
            input_ids = encodings.input_ids.to(model.device)
            attention_mask = encodings.attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())
    import numpy as np
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    return {"perplexity": round(float(perplexity), 2)}

def main():
    parser = argparse.ArgumentParser(description="LLM benchmarking util")
    parser.add_argument("--ckpt", required=True, help="Model name or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (if any)")
    parser.add_argument("--ppl_dataset", default=None, help="HF dataset for perplexity")
    parser.add_argument("--ppl_split", default="validation[:1024]", help="Dataset split for PPL")
    parser.add_argument("--prompt", default="Hello", help="Prompt for speed test")
    parser.add_argument("--out", default="bench.json", help="Where to save JSON")
    args = parser.parse_args()

    model, tok = load_model(args.ckpt, args.adapter)
    metrics = {
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    metrics.update(memory_metrics(model))
    metrics.update(latency_metrics(model, tok, args.prompt))
    if args.ppl_dataset:
        metrics.update(compute_perplexity(model, tok, args.ppl_dataset, args.ppl_split))
    print(json.dumps(metrics, indent=2))
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()