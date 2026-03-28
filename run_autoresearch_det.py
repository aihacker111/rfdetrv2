"""
run_autoresearch.py — Autonomous research loop for RF-DETRv2 fine-tuning.

Replaces aider entirely. Does the loop itself:
  1. Ask LLM for next experiment idea
  2. Apply changes to finetune.py
  3. Run python finetune.py
  4. Read results
  5. Keep or discard
  6. Repeat N times

Usage:
    export OPENROUTER_API_KEY=sk-or-xxx
    export DATASET_DIR=/path/to/data
    export COCO_WEIGHTS=/path/to/weight.pth
    python run_autoresearch.py --experiments 10
"""

import os, sys, json, re, shutil, subprocess, time, argparse
from pathlib import Path
from datetime import datetime
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL              = "openai/gpt-4.1-mini"   # cheap + smart, change if needed
FINETUNE_FILE      = "finetune.py"
RESULTS_FILE       = "results.tsv"
LOG_FILE           = "run.log"
TIMEOUT_FACTOR     = 2.5   # kill if > 2.5x baseline time


def ask_llm(messages: list, max_tokens=1500) -> str:
    """Call OpenRouter API."""
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


def read_file(path: str) -> str:
    return Path(path).read_text(errors="ignore")


def write_file(path: str, content: str):
    Path(path).write_text(content)


def git_commit(msg: str) -> str:
    subprocess.run(["git", "add", FINETUNE_FILE], capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)


def run_training(timeout: float = None) -> tuple[bool, float]:
    """Run finetune.py, return (success, elapsed_seconds)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, FINETUNE_FILE],
            stdout=open(LOG_FILE, "w"),
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        return proc.returncode == 0, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] Killed after {elapsed:.0f}s")
        return False, elapsed


def parse_metrics() -> dict:
    """Parse val_mAP and peak_vram_mb from run.log."""
    metrics = {"val_mAP": 0.0, "val_mAP50": 0.0, "peak_vram_mb": 0.0}
    try:
        log = read_file(LOG_FILE)
        for line in log.splitlines():
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


def get_next_experiment(history: list, current_finetune: str) -> dict:
    """Ask LLM for next hyperparameter change. Returns {description, new_code}."""

    history_str = "\n".join(
        f"  exp{i+1}: {h['description']} → mAP={h['val_mAP']:.4f} ({h['status']})"
        for i, h in enumerate(history)
    ) or "  (no experiments yet — this is the baseline)"

    messages = [
        {"role": "system", "content": (
            "You are an expert ML researcher optimizing RF-DETRv2 fine-tuning on small datasets. "
            "You modify hyperparameters in finetune.py to maximize val_mAP. "
            "Reply ONLY with valid JSON: {\"description\": \"...\", \"new_code\": \"...(full finetune.py content)...\"}. "
            "No markdown, no explanation outside JSON."
        )},
        {"role": "user", "content": f"""
Experiment history so far:
{history_str}

Current finetune.py:
```python
{current_finetune}
```

Pick ONE hyperparameter change likely to improve val_mAP for small-data fine-tuning.
Focus on: LR, freeze_encoder, loss weights, scheduler, prototype alignment.
Return the full modified finetune.py as new_code.
"""}
    ]

    response = ask_llm(messages, max_tokens=3000)

    # Parse JSON from response
    # Strip markdown code blocks if present
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*", "", response)
    parsed = json.loads(response)
    return parsed


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY env var")
        sys.exit(1)

    # Init results.tsv
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(
            "commit\tval_mAP\tval_mAP50\tmemory_gb\tstatus\tdescription\n"
        )

    # Create branch
    tag = datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"
    subprocess.run(["git", "checkout", "-b", branch], capture_output=True)
    print(f"\n{'='*60}")
    print(f"  AutoResearch RF-DETRv2 | branch: {branch}")
    print(f"  Model: {MODEL} | Max experiments: {args.experiments}")
    print(f"{'='*60}\n")

    history = []
    baseline_time = None
    best_mAP = 0.0

    for exp_num in range(1, args.experiments + 1):
        print(f"\n── Experiment {exp_num}/{args.experiments} ──────────────────")

        # ── Get next experiment idea from LLM (skip for baseline) ──
        current_code = read_file(FINETUNE_FILE)

        if exp_num == 1:
            description = "baseline — no changes"
            print(f"  Idea: {description}")
        else:
            print("  Asking LLM for next idea...")
            try:
                result = get_next_experiment(history, current_code)
                description = result["description"]
                new_code    = result["new_code"]
                write_file(FINETUNE_FILE, new_code)
                print(f"  Idea: {description}")
            except Exception as e:
                print(f"  LLM error: {e}. Skipping.")
                continue

        # ── Commit ──
        commit = git_commit(f"experiment: {description}")
        print(f"  Commit: {commit}")

        # ── Run training ──
        timeout = (baseline_time * TIMEOUT_FACTOR) if baseline_time else None
        print(f"  Training... (timeout={f'{timeout:.0f}s' if timeout else 'none'})")
        success, elapsed = run_training(timeout=timeout)

        if exp_num == 1:
            baseline_time = elapsed

        # ── Parse results ──
        metrics = parse_metrics()
        val_mAP = metrics["val_mAP"]

        if not success or val_mAP == 0.0:
            print(f"  ✗ CRASH — {elapsed:.0f}s")
            append_results(commit, metrics, "crash", description)
            history.append({"description": description, "val_mAP": 0.0, "status": "crash"})
            git_reset()
            continue

        print(f"  val_mAP={val_mAP:.4f}  mAP50={metrics['val_mAP50']:.4f}  "
              f"VRAM={metrics['peak_vram_mb']/1024:.1f}GB  time={elapsed:.0f}s")

        # ── Keep or discard ──
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            status = "keep"
            print(f"  ✓ KEEP (+{val_mAP - (history[-1]['val_mAP'] if history else 0):.4f})")
        else:
            status = "discard"
            print(f"  ✗ DISCARD (no improvement)")
            git_reset()

        append_results(commit, metrics, status, description)
        history.append({"description": description, "val_mAP": val_mAP, "status": status})

        # ── Plot progress ──
        subprocess.run([sys.executable, "plot_progress.py"], capture_output=True)

    # ── Final summary ──
    best = max(history, key=lambda h: h["val_mAP"])
    baseline_mAP = history[0]["val_mAP"] if history else 0
    improvement = best["val_mAP"] - baseline_mAP
    pct = improvement / baseline_mAP * 100 if baseline_mAP > 0 else 0

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH COMPLETE ({len(history)} experiments)")
    print(f"  Baseline mAP : {baseline_mAP:.4f}")
    print(f"  Best mAP     : {best['val_mAP']:.4f}")
    print(f"  Improvement  : {improvement:+.4f} ({pct:+.1f}%)")
    print(f"  Best config  : {best['description']}")
    print(f"  Charts       : progress.png, progress_detail.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()