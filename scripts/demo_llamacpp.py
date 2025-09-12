"""Demo (llama.cpp): merge LoRA -> GGUF -> quantize -> bench (final).

This is the base demo pipeline using llama.cpp tools:
1) Merge LoRA into base (HF format)
2) Convert merged model to GGUF
3) Build llama.cpp (best-effort; uses CUDA if available, falls back to CPU)
4) Create a small eval slice
5) Quantize to selected variants (e.g., Q8_0, Q6_K)
6) Run llama-perplexity and llama-bench
7) Plot a 4-panel compression figure and save JSON metrics

Usage:
  python scripts/demo_llamacpp.py \
      --ckpt meta-llama/Meta-Llama-3-8B-Instruct \
      --adapter outputs/naive_run/lora_adapter \
      --variants Q8_0 Q6_K

Artifacts:
  outputs/demo/{full-f16.gguf, model_<VARIANT>.gguf, eval_slice.txt, metrics.json, demo.png}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# Avoid optional torchvision import paths inside transformers
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def ensure_py_deps():
    """Install pinned Python deps (matching environment.yml) to avoid ABI mismatches."""
    import importlib.util
    pin = {
        "transformers": "4.52.4",
        "datasets": "3.6.0",
        "peft": "0.15.2",
        "sentencepiece": "0.2.0",
        "huggingface_hub": "0.32.4",
    }
    to_install = []
    for pkg, ver in pin.items():
        if importlib.util.find_spec(pkg) is None:
            to_install.append(f"{pkg}=={ver}")
    if to_install:
        print("[deps] Installing:", ", ".join(to_install))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[run]", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def ensure_hf_login_interactive() -> bool:
    try:
        from huggingface_hub import whoami, login
        try:
            whoami()
            return True
        except Exception:
            pass
        token = input("Enter Hugging Face token (hf_..., leave blank to skip): ").strip()
        if token:
            login(token=token, add_to_git_credential=True)
            return True
    except Exception as e:
        print("[warn] HF login failed:", e)
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Base HF model repo or dir")
    p.add_argument("--adapter", default=None, help="LoRA/PEFT adapter directory (optional)")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "demo"), help="Output directory")
    p.add_argument("--llama_cpp_dir", default=str(REPO_ROOT / "llama.cpp"), help="Path to llama.cpp repo clone")
    p.add_argument("--variants", nargs="*", default=["Q8_0", "Q4_K_M", "Q3_K_S"], help="Quantization variants to build")
    p.add_argument("--threads", type=int, default=8, help="Threads for llama-perplexity")
    p.add_argument("--ppl_rows", type=int, default=100, help="Rows for eval_slice.txt (tiny slice)")
    p.add_argument("--cuda_arch", default="90", help="CUDA SM arch for build (e.g., 80, 86, 89, 90)")
    p.add_argument("--rebuild", action="store_true", help="Force reconfigure/rebuild llama.cpp")
    p.add_argument("--force_cpu", action="store_true", help="Run evals on CPU only (-ngl 0)")
    return p.parse_args()


def step_merge(ckpt: str, adapter: str, out_dir: Path) -> Path:
    merged_dir = out_dir / "merged_model"
    if merged_dir.exists() and any(merged_dir.iterdir()):
        print("[merge] Using existing merged model:", merged_dir)
        return merged_dir
    print("[merge] Merging LoRA into base model ->", merged_dir)
    # Call merge_lora.py via Python to avoid new processes parsing complexity
    import importlib.util
    spec = importlib.util.spec_from_file_location("merge_lora", str(THIS_DIR / "merge_lora.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    # Reuse its main by invoking via subprocess-like args
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["merge_lora.py", "--ckpt", ckpt, "--adapter", adapter, "--out", str(merged_dir)]
        mod.main()  # type: ignore
    finally:
        sys.argv = sys_argv_backup
    return merged_dir


def ensure_llama_cpp(llama_cpp_dir: Path) -> Path:
    if not llama_cpp_dir.exists():
        print("[llama.cpp] Cloning repo ->", llama_cpp_dir)
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)])

    # Python requirements for conversion
    req = llama_cpp_dir / "requirements.txt"
    if req.exists():
        try:
            # Avoid installing torch/torchvision from llama.cpp requirements in GPU images that already ship them
            # by filtering out torch/torchvision lines.
            filtered = []
            with open(req, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(("torch", "torchvision")):
                        continue
                    filtered.append(line)
            tmp = llama_cpp_dir / "requirements.filtered.txt"
            tmp.write_text("".join(filtered), encoding="utf-8")
            run([sys.executable, "-m", "pip", "install", "-r", str(tmp)])
        except subprocess.CalledProcessError as e:
            print("[warn] pip install requirements failed:", e)
    return llama_cpp_dir


def download_model_if_needed(model_id: str, out_dir: Path) -> Path:
    """Download HF model to local dir if it's a model ID, return the path."""
    if Path(model_id).exists():
        return Path(model_id)
    
    # Download to a subdirectory
    local_dir = out_dir / "downloaded_model"
    if local_dir.exists() and any(local_dir.iterdir()):
        print("[download] Using existing downloaded model:", local_dir)
        return local_dir
    
    print(f"[download] Downloading {model_id} -> {local_dir}")
    try:
        from huggingface_hub import snapshot_download
        hf_token = os.getenv("HF_TOK") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        local_dir.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_dir_use_symlinks=False, token=hf_token)
        print(f"[download] Saved to {resolved}")
        return Path(resolved)
    except Exception as e:
        print(f"[warn] Download failed: {e}")
        raise


def convert_to_gguf(source_model: str | Path, llama_cpp_dir: Path, out_dir: Path) -> Path:
    gguf = out_dir / "full-f16.gguf"
    if gguf.exists():
        print("[convert] Using existing GGUF:", gguf)
        return gguf
    
    # Ensure we have a local directory
    local_model_dir = download_model_if_needed(str(source_model), out_dir)
    
    print("[convert] Converting HF -> GGUF:", gguf)
    env = os.environ.copy()
    # Some HF models require an auth token.
    if env.get("HF_TOK"):
        env["HUGGING_FACE_HUB_TOKEN"] = env.get("HF_TOK")
        env["HUGGINGFACEHUB_API_TOKEN"] = env.get("HF_TOK")
    try:
        subprocess.run([sys.executable, str(llama_cpp_dir / "convert_hf_to_gguf.py"), str(local_model_dir), "--outfile", str(gguf)],
                       check=True, text=True, capture_output=True, env=env)
    except subprocess.CalledProcessError as e:
        print("[warn] convert_hf_to_gguf failed. If the model is gated, run 'huggingface-cli login' or set HF_TOK.")
        print(e.stderr)
        raise
    return gguf


def build_llama_cpp(llama_cpp_dir: Path, cuda_arch: str = "90", force_rebuild: bool = False) -> Path:
    build_dir = llama_cpp_dir / "build"
    bin_llama_bench = build_dir / "bin" / "llama-bench"
    if force_rebuild and build_dir.exists():
        print("[build] Forcing rebuild: removing", build_dir)
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    print("[build] Configuring CMake ...")
    # Ensure cmake exists
    cmake_bin = shutil.which("cmake")
    if not cmake_bin:
        print("[build] 'cmake' not found. Attempting to install via apt...")
        try:
            run(["apt-get", "update"])
            run(["apt-get", "install", "-y", "cmake"])
            cmake_bin = shutil.which("cmake")
        except Exception as e:
            print("[warn] apt-get install cmake failed:", e)
        if not cmake_bin:
            print("[build] Falling back to pip installing 'cmake' wheel...")
            run([sys.executable, "-m", "pip", "install", "cmake"]) 
            cmake_bin = shutil.which("cmake") or sys.executable + " -m cmake"

    # Prefer CUDA build if nvcc present, else CPU
    nvcc = shutil.which("nvcc")
    cmake_cmd = [cmake_bin, ".."] if " " not in cmake_bin else ["bash", "-lc", f"{cmake_bin} .."]
    if nvcc:
        cmake_cmd += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
    else:
        cmake_cmd += ["-DGGML_CUDA=OFF"]
    # Disable CURL to avoid failing when libcurl dev is missing in minimal images
    cmake_cmd += ["-DLLAMA_CURL=OFF"]
    try:
        run(cmake_cmd, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print("[warn] CUDA CMake configure failed, falling back to CPU-only. Details:\n", e.stderr)
        run([cmake_bin, "..", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"], cwd=build_dir)
    print("[build] Building ... (may take a while)")
    build_cmd = ["cmake", "--build", ".", f"-j{os.cpu_count() or 4}"]
    if " " in cmake_bin:
        build_cmd = ["bash", "-lc", f"{cmake_bin.split(' -m ')[0]} -m cmake --build . -j{os.cpu_count() or 4}"]
    run(build_cmd, cwd=build_dir)
    return build_dir


def _strip_ansi(text: str) -> str:
    try:
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    except Exception:
        return text


def ensure_eval_slice(out_dir: Path, ppl_rows: int) -> Path:
    slice_path = out_dir / "eval_slice.txt"
    if slice_path.exists():
        print("[eval] Using existing slice:", slice_path)
        return slice_path
    print("[eval] Creating eval slice ->", slice_path)
    # Reuse eval_slicer.py but adjust rows
    from datasets import load_dataset
    from itertools import islice
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    start = 100_000
    end = start + ppl_rows
    with open(slice_path, "w", encoding="utf-8") as f:
        for row in islice(ds, start, end):
            content = row.get("content", "").strip()
            if content:
                f.write(content + "\n")
    return slice_path


def quantize(build_dir: Path, input_gguf: Path, out_dir: Path, variant: str) -> Path:
    def _is_valid_gguf(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
            return magic == b"GGUF"
        except Exception:
            return False

    out = out_dir / f"model_{variant}.gguf"
    if out.exists():
        if _is_valid_gguf(out) and out.stat().st_size > 0:
            print(f"[quantize] Using existing {variant} -> {out}")
            return out
        else:
            print(f"[quantize] Detected invalid/corrupt GGUF for {variant}, re-quantizing ...")
            try:
                out.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
    cmd = [str(build_dir / "bin" / "llama-quantize"), str(input_gguf), str(out), variant]
    run(cmd)
    # Validate after quantization
    try:
        with open(out, "rb") as f:
            if f.read(4) != b"GGUF":
                raise RuntimeError("quantize produced non-GGUF output")
    except Exception as e:
        print(f"[warn] Quantized file validation failed for {variant}: {e}")
    return out


def run_llama_perplexity(build_dir: Path, model_path: Path, slice_path: Path, threads: int, force_cpu: bool = False) -> Optional[float]:
    cmd = [str(build_dir / "bin" / "llama-perplexity"), "-m", str(model_path), "-f", str(slice_path), "-n", "128", "-t", str(threads)]
    if force_cpu or os.getenv("LLAMACPP_NGL", "").strip() == "0":
        cmd += ["-ngl", "0"]
    try:
        res = run(cmd)
    except subprocess.CalledProcessError as e:
        print("[warn] llama-perplexity failed:", e.stderr)
        # Fallback: retry on CPU to avoid aborting the pipeline
        try:
            print("[warn] Retrying llama-perplexity on CPU (-ngl 0)")
            res = run(cmd + ["-ngl", "0"])
        except subprocess.CalledProcessError as e2:
            print("[warn] llama-perplexity CPU retry failed:", e2.stderr)
            return None
    # Dump raw output for visibility
    txt = _strip_ansi((res.stdout or "") + (res.stderr or ""))
    try:
        dump_path = Path(str(model_path) + ".perplexity.txt")
        dump_path.write_text(txt, encoding="utf-8")
        print("[perplexity output]\n" + txt)
    except Exception:
        pass
    # Prefer the final estimate that llama-perplexity prints, e.g.:
    #   "Final estimate: PPL = 3.7993 +/- 0.03461"
    m_final = re.search(r"Final\s+estimate\s*:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE)
    if m_final:
        return float(m_final.group(1))
    # Some versions may print "ppl = <val>" explicitly
    m_ppl = re.search(r"\bppl\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE)
    if m_ppl:
        return float(m_ppl.group(1))
    # Avoid capturing "perplexity: <seconds per pass>"; only accept '=' form
    m_eq = re.search(r"\bperplexity\b\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE)
    if m_eq:
        return float(m_eq.group(1))
    # Last resort: look for a number following the word 'perplexity' on the same line,
    # but ignore patterns with 'seconds per pass'
    for line in txt.splitlines()[::-1]:
        if re.search(r"perplexity", line, re.IGNORECASE) and not re.search(r"seconds\s+per\s+pass", line, re.IGNORECASE):
            m_any = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if m_any:
                try:
                    return float(m_any.group(1))
                except Exception:
                    pass
    return None


def run_llama_bench(build_dir: Path, model_path: Path, threads: int = 8, force_cpu: bool = False) -> Optional[float]:
    cmd_base = [str(build_dir / "bin" / "llama-bench"), "-m", str(model_path), "-n", "512", "-t", str(threads)]
    if force_cpu or os.getenv("LLAMACPP_NGL", "").strip() == "0":
        cmd_base += ["-ngl", "0"]

    def _parse_tokps(text: str) -> Optional[float]:
        txt_norm = text.replace(",", "")
        # tokens/sec variants (number-first)
        patterns = [
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|t/s|tokens/s|tokens/sec|token/s|tokens per second)",
            r"throughput[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|t/s|tokens/s|tokens/sec)",
            r"decod[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|t/s|tokens/s|tokens/sec)",
        ]
        vals: List[float] = []
        for pat in patterns:
            for m in re.findall(pat, txt_norm, flags=re.IGNORECASE):
                try:
                    vals.append(float(m))
                except Exception:
                    pass
        if vals:
            return max(vals)
        # ASCII table row: "| ... | tg512 |   172.00 ± 0.06 |"
        tg_matches = re.findall(r"\|[^\n]*\|\s*tg\d+\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*±", txt_norm, flags=re.IGNORECASE)
        if tg_matches:
            try:
                return max(float(x) for x in tg_matches)
            except Exception:
                pass
        # label-first variants e.g. "tok/s: 123.4" or "tokens/s = 123.4"
        for m in re.findall(r"(?:tok/s|t/s|tokens/s|tokens/sec|token/s|tokens per second)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", txt_norm, flags=re.IGNORECASE):
            try:
                vals.append(float(m))
            except Exception:
                pass
        if vals:
            return max(vals)
        # ms/tok → invert
        ms_vals: List[float] = []
        for m in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*ms/(?:tok|token)", txt_norm, flags=re.IGNORECASE):
            try:
                ms_vals.append(float(m))
            except Exception:
                pass
        if ms_vals:
            best_ms = min(ms_vals)
            if best_ms > 0:
                return 1000.0 / best_ms
        # generic avg capture
        m = re.search(r"avg\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:tok/s|t/s|tokens/s)?", txt_norm, flags=re.IGNORECASE)
        return float(m.group(1)) if m else None

    # Prefer CSV output if supported to get a stable field
    tried_csv = False
    for attempt in range(2):
        cmd = list(cmd_base)
        if attempt == 0:
            cmd += ["-of", "csv"]
            tried_csv = True
        try:
            res = run(cmd)
            txt = _strip_ansi((res.stdout or "") + (res.stderr or ""))
            # Save raw output next to the model for debugging and print it
            try:
                log_path = Path(str(model_path) + ".bench.log")
                log_path.write_text(txt, encoding="utf-8")
                print("[bench output]\n" + txt)
            except Exception:
                pass
            # If CSV, pick the column that contains any 'tok/s' token in header
            if tried_csv and ("," in res.stdout or ";" in res.stdout or "\t" in res.stdout):
                lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
                if len(lines) >= 2:
                    # detect delimiter
                    delim = ","
                    for d in [",", ";", "\t"]:
                        if d in lines[0] and d in lines[-1]:
                            delim = d
                            break
                    header = [h.strip().lower() for h in lines[0].split(delim)]
                    row = [c.strip() for c in lines[-1].split(delim)]
                    # Find decode tokens/sec style columns
                    target_idx = -1
                    for idx, name in enumerate(header):
                        if ("tok/s" in name) or ("tokens/s" in name) or ("t/s" in name and "decode" in name):
                            target_idx = idx
                    if target_idx >= 0 and target_idx < len(row):
                        try:
                            return float(row[target_idx].replace(",", ""))
                        except Exception:
                            pass
                    # Fallback: take the maximum float from the last data row
                    nums: List[float] = []
                    for cell in row:
                        try:
                            nums.append(float(cell.replace(",", "")))
                        except Exception:
                            pass
                    if nums:
                        return max(nums)
            # Fallback to robust text parsing
            val = _parse_tokps(txt)
            if val is not None:
                return val
        except subprocess.CalledProcessError as e:
            # retry without CSV flag or on CPU if not already forced
            if attempt == 0:
                continue
            print("[warn] llama-bench failed:", e.stderr)
            if "-ngl" not in cmd_base:
                try:
                    print("[warn] Retrying llama-bench on CPU (-ngl 0)")
                    res = run(cmd_base + ["-ngl", "0"])
                    txt = (res.stdout or "") + (res.stderr or "")
                    return _parse_tokps(txt)
                except subprocess.CalledProcessError as e2:
                    print("[warn] llama-bench CPU retry failed:", e2.stderr)
            return None
    return None


def make_plot(points: List[Dict[str, object]], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _num(x, fallback=0.0):
        try:
            return float(x) if x is not None else float(fallback)
        except Exception:
            return float(fallback)

    names = [p["name"] for p in points]
    sizes = [_num(p["size_gb"], 0.0) for p in points]
    tokps = [_num(p["tokps"], 0.0) for p in points]
    ppls  = [_num(p["ppl"], float("nan")) for p in points]

    # Choose best: high tokens/sec, low ppl
    best_idx, best_score = 0, -1e9
    for i, (t, q) in enumerate(zip(tokps, ppls)):
        score = (t or 0.0) - 10.0 * ((q or 10.0))
        if score > best_score:
            best_idx, best_score = i, score

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    ax = axs[0, 0]
    ax.scatter(tokps, ppls, c="#4e79a7")
    for i, label in enumerate(names):
        ax.annotate(label, (tokps[i] or 0.0, ppls[i] or 0.0), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.scatter([tokps[best_idx]], [ppls[best_idx]], marker="*", s=220, color="#f28e2b", edgecolor="black", zorder=3)
    ax.set_xlabel("tokens/sec")
    ax.set_ylabel("perplexity")
    ax.set_title("Speed vs Perplexity (llama.cpp)")
    ax.grid(True, alpha=0.3)

    ax = axs[0, 1]
    ax.scatter(sizes, tokps, c="#59a14f")
    for i, label in enumerate(names):
        ax.annotate(label, (sizes[i] or 0.0, tokps[i] or 0.0), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("model size (GB)")
    ax.set_ylabel("tokens/sec")
    ax.set_title("Speed vs Model Size (llama.cpp)")
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    ax.bar(names, tokps, color="#e15759")
    for i, t in enumerate(tokps):
        ax.text(i, (t or 0.0) + 0.02, f"{sizes[i]:.2f} GB", ha="center", fontsize=8)
    ax.set_ylabel("tokens/sec")
    ax.set_title("Inference Speed by Variant")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axs[1, 1]
    ax.bar(names, ppls, color="#b07aa1")
    for i, v in enumerate(ppls):
        try:
            ax.text(i, (v if v == v else 0.0) + 0.02, f"{float(v):.2f}", ha="center", fontsize=8)
        except Exception:
            ax.text(i, 0.02, "nan", ha="center", fontsize=8)
    ax.set_ylabel("perplexity")
    ax.set_title("Perplexity by Variant")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def bytes_to_gb(path: Path) -> float:
    try:
        return round(path.stat().st_size / 1e9, 2)
    except Exception:
        return 0.0


def main():
    args = parse_args()
    ensure_py_deps()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # llama.cpp checkout + requirements
    llama_cpp_dir = ensure_llama_cpp(Path(args.llama_cpp_dir))

    # Convert to GGUF: if adapter provided, merge first; else convert base ckpt directly
    if args.adapter:
        try:
            merged_dir = step_merge(args.ckpt, args.adapter, out_dir)
        except Exception as e:
            print("[warn] Merge failed, trying HF login:", e)
            if ensure_hf_login_interactive():
                merged_dir = step_merge(args.ckpt, args.adapter, out_dir)
            else:
                raise
        source_model = merged_dir
    else:
        source_model = args.ckpt

    gguf_path = convert_to_gguf(source_model, llama_cpp_dir, out_dir)

    # Build llama.cpp (CUDA if possible)
    build_dir = build_llama_cpp(llama_cpp_dir, args.cuda_arch, args.rebuild)

    # Eval slice
    slice_path = ensure_eval_slice(out_dir, args.ppl_rows)

    points: List[Dict[str, object]] = []

    # Also record full-f16 "as is" metrics if llama-bench supports it (may be huge for 8B)
    base_name = "full-f16"
    base_size_gb = bytes_to_gb(gguf_path)
    ppl_base = run_llama_perplexity(build_dir, gguf_path, slice_path, args.threads, args.force_cpu)
    tokps_base = run_llama_bench(build_dir, gguf_path, args.threads, args.force_cpu)
    points.append({"name": base_name, "size_gb": base_size_gb, "ppl": ppl_base, "tokps": tokps_base})

    # Quantize variants
    for v in args.variants:
        q_path = quantize(build_dir, gguf_path, out_dir, v)
        size_gb = bytes_to_gb(q_path)
        ppl = run_llama_perplexity(build_dir, q_path, slice_path, args.threads, args.force_cpu)
        tokps = run_llama_bench(build_dir, q_path, args.threads, args.force_cpu)
        points.append({"name": v, "size_gb": size_gb, "ppl": ppl, "tokps": tokps})

    # Save metrics + plot
    metrics = {
        "ckpt": args.ckpt,
        "adapter": args.adapter,
        "variants": args.variants,
        "points": points,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    make_plot(points, out_dir / "demo.png")
    print(json.dumps(metrics, indent=2))
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()


