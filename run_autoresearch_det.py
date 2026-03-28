"""
run_autoresearch.py — Autonomous research loop for RF-DETRv2 fine-tuning.

Replaces aider entirely. Does the loop itself:
  1. Ask LLM for next experiment idea
  2. Apply changes to finetune.py
  3. Run training (single-GPU or torchrun multi-GPU)
  4. Read results
  5. Keep or discard
  6. Repeat N times

Usage (single GPU):
    python run_autoresearch.py --experiments 10

Usage (multi-GPU, e.g. 4 GPUs):
    python run_autoresearch.py --experiments 10 --gpus 4

Env vars:
    OPENROUTER_API_KEY, DATASET_DIR, COCO_WEIGHTS
"""

import os, sys, json, re, subprocess, time, argparse
from pathlib import Path
from datetime import datetime
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL              = "openai/gpt-4.1-mini"
FINETUNE_FILE      = "finetune.py"
RESULTS_FILE       = "results.tsv"
LOG_FILE           = "run.log"
TIMEOUT_FACTOR     = 2.5
NUM_GPUS           = 1   # overridden by --gpus


# ── GPU utils ─────────────────────────────────────────────────────────────────

def detect_gpus() -> int:
    """Auto-detect number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
        return max(count, 1)
    except Exception:
        return 1


def build_train_cmd(num_gpus: int) -> list:
    """Build training command: python (1 GPU) or torchrun (multi-GPU)."""
    if num_gpus <= 1:
        return [sys.executable, FINETUNE_FILE]
    else:
        return [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            FINETUNE_FILE,
        ]


# ── LLM ───────────────────────────────────────────────────────────────────────

def ask_llm(messages: list, max_tokens=3000) -> str:
    data = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"].strip()


def get_next_experiment(history: list, current_finetune: str, num_gpus: int) -> dict:
    history_str = "\n".join(
        f"  exp{i+1}: {h['description']} → mAP={h['val_mAP']:.4f} ({h['status']})"
        for i, h in enumerate(history)
    ) or "  (no experiments yet — this is the baseline)"

    messages = [
        {"role": "system", "content": (
            "You are an expert ML researcher optimizing RF-DETRv2 fine-tuning on small datasets. "
            "You modify hyperparameters in finetune.py to maximize val_mAP. "
            f"Training runs on {num_gpus} GPU(s). "
            f"{'With multi-GPU, effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS * num_gpus, so scale LR accordingly.' if num_gpus > 1 else ''}"
            "Reply ONLY with valid JSON: "
            "{\"description\": \"short description\", \"new_code\": \"full finetune.py content\"}. "
            "No markdown fences, no explanation outside JSON."
        )},
        {"role": "user", "content": (
            f"Experiment history:\n{history_str}\n\n"
            f"Current finetune.py:\n{current_finetune}\n\n"
            "Pick ONE hyperparameter change to improve val_mAP.\n"
            "Focus: LR, freeze_encoder, loss weights, scheduler, prototype alignment.\n"
            "Return full modified finetune.py as new_code."
        )}
    ]

    response = ask_llm(messages)
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```python\s*", "", response)
    response = re.sub(r"```\s*", "", response)
    response = response.encode("utf-8", errors="replace").decode("utf-8")

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        desc = re.search(r'"description"\s*:\s*"([^"]+)"', response)
        code = re.search(r'"new_code"\s*:\s*"(.*?)"\s*[,}]', response, re.DOTALL)
        if not desc or not code:
            raise ValueError(f"Cannot parse LLM JSON: {response[:200]}")
        parsed = {
            "description": desc.group(1),
            "new_code": code.group(1).replace("\\n", "\n").replace('\\"', '"')
        }
    return parsed


# ── Git ───────────────────────────────────────────────────────────────────────

def git_commit(msg: str) -> str:
    subprocess.run(["git", "add", FINETUNE_FILE], capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], capture_output=True)
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(num_gpus: int, timeout: float = None) -> tuple:
    cmd = build_train_cmd(num_gpus)
    t0  = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=open(LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode == 0, time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] Killed after {elapsed:.0f}s")
        return False, elapsed


def parse_metrics() -> dict:
    metrics = {"val_mAP": 0.0, "val_mAP50": 0.0, "peak_vram_mb": 0.0}
    try:
        for line in Path(LOG_FILE).read_text(errors="ignore").splitlines():
            for key in metrics:
                m = re.match(rf"^{key}:\s+([\d.]+)", line)
                if m:
                    metrics[key] = float(m.group(1))
    except Exception:
        pass
    return metrics


def append_results(commit, metrics, status, description):
    mem_gb = round(metrics["peak_vram_mb"] / 1024, 1)
    row = (f"{commit}\t{metrics['val_mAP']:.6f}\t{metrics['val_mAP50']:.6f}"
           f"\t{mem_gb}\t{status}\t{description}\n")
    with open(RESULTS_FILE, "a") as f:
        f.write(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MODEL, NUM_GPUS

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--model",       default=MODEL)
    parser.add_argument("--cuda-devices", default=None,
                        help="CUDA_VISIBLE_DEVICES, e.g. 0,1 or 0")
    parser.add_argument("--gpus",        type=int, default=0,
                        help="Number of GPUs (0=auto-detect)")
    args = parser.parse_args()

    MODEL = args.model

    # Set CUDA_VISIBLE_DEVICES before anything GPU-related
    cuda_devices = args.cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        visible_count = len(cuda_devices.split(","))
        NUM_GPUS = args.gpus if args.gpus > 0 else visible_count
    else:
        NUM_GPUS = args.gpus if args.gpus > 0 else detect_gpus()

    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY"); sys.exit(1)

    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(
            "commit\tval_mAP\tval_mAP50\tmemory_gb\tstatus\tdescription\n"
        )

    tag    = datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"
    r = subprocess.run(["git", "checkout", "-b", branch], capture_output=True)
    if r.returncode != 0:
        subprocess.run(["git", "checkout", branch], capture_output=True)

    train_cmd_str = " ".join(build_train_cmd(NUM_GPUS))
    print(f"\n{'='*60}")
    print(f"  AutoResearch RF-DETRv2")
    print(f"  Branch      : {branch}")
    print(f"  LLM Model   : {MODEL}")
    print(f"  GPUs        : {NUM_GPUS} → {train_cmd_str}")
    print(f"  Experiments : {args.experiments}")
    print(f"{'='*60}\n")

    history      = []
    baseline_time = None
    best_mAP     = 0.0

    for exp_num in range(1, args.experiments + 1):
        print(f"\n── Experiment {exp_num}/{args.experiments} ──────────────────")

        current_code = Path(FINETUNE_FILE).read_text(errors="ignore")

        if exp_num == 1:
            description = "baseline — no changes"
            print(f"  Idea: {description}")
        else:
            print("  Asking LLM for next idea...")
            try:
                result      = get_next_experiment(history, current_code, NUM_GPUS)
                description = result["description"]
                Path(FINETUNE_FILE).write_text(result["new_code"])
                print(f"  Idea: {description}")
            except Exception as e:
                print(f"  LLM error: {e}. Skipping.")
                continue

        commit  = git_commit(f"experiment: {description}")
        timeout = (baseline_time * TIMEOUT_FACTOR) if baseline_time else None
        print(f"  Commit: {commit} | GPUs: {NUM_GPUS} | "
              f"Timeout: {f'{timeout:.0f}s' if timeout else 'none'}")
        print(f"  Training...")

        success, elapsed = run_training(NUM_GPUS, timeout=timeout)
        if exp_num == 1:
            baseline_time = elapsed

        metrics = parse_metrics()
        val_mAP = metrics["val_mAP"]

        if not success or val_mAP == 0.0:
            print(f"  ✗ CRASH ({elapsed:.0f}s) — check run.log")
            append_results(commit, metrics, "crash", description)
            history.append({"description": description, "val_mAP": 0.0, "status": "crash"})
            git_reset()
            continue

        print(f"  val_mAP={val_mAP:.4f}  mAP50={metrics['val_mAP50']:.4f}  "
              f"VRAM={metrics['peak_vram_mb']/1024:.1f}GB  time={elapsed:.0f}s")

        prev_mAP = history[-1]["val_mAP"] if history else 0.0
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            status   = "keep"
            print(f"  ✓ KEEP  (Δ={val_mAP - prev_mAP:+.4f})")
        else:
            status = "discard"
            print(f"  ✗ DISCARD (Δ={val_mAP - prev_mAP:+.4f})")
            git_reset()

        append_results(commit, metrics, status, description)
        history.append({"description": description, "val_mAP": val_mAP, "status": status})
        subprocess.run([sys.executable, "plot_progress.py"], capture_output=True)

    # ── Final summary ──
    if not history:
        print("No experiments completed."); return

    best         = max(history, key=lambda h: h["val_mAP"])
    baseline_mAP = history[0]["val_mAP"]
    improvement  = best["val_mAP"] - baseline_mAP
    pct          = improvement / baseline_mAP * 100 if baseline_mAP > 0 else 0

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH COMPLETE ({len(history)} experiments)")
    print(f"  GPUs used    : {NUM_GPUS}")
    print(f"  Baseline mAP : {baseline_mAP:.4f}")
    print(f"  Best mAP     : {best['val_mAP']:.4f}")
    print(f"  Improvement  : {improvement:+.4f} ({pct:+.1f}%)")
    print(f"  Best config  : {best['description']}")
    print(f"  Charts       : progress.png, progress_detail.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()