#!/usr/bin/env python
"""Straight speed test: fixed prompt, fixed output length, same endpoint."""
import time
import requests

BASE_URL = "http://127.0.0.1:11434"
MODELS = ["qwen3.8:27b", "gemma4:26b"]
PROMPT = ("Write a detailed, technical explanation of how a bicycle's rear derailleur "
          "gear system works, including how it moves the chain between sprockets.")
NUM_PREDICT = 300
RUNS = 3


def call(model, warm=False):
    t0 = time.monotonic()
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "options": {"num_predict": NUM_PREDICT, "temperature": 0.7},
        },
        timeout=180,
    )
    wall_s = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    return wall_s, data


def ns_to_s(ns):
    return (ns or 0) / 1e9


for model in MODELS:
    print(f"\n=== {model} ===")
    wall_s, data = call(model, warm=True)
    load_s = ns_to_s(data.get("load_duration"))
    print(f"  warm-up: wall={wall_s:.2f}s  load_duration={load_s:.2f}s")

    tok_rates = []
    for i in range(RUNS):
        wall_s, data = call(model)
        eval_count = data.get("eval_count", 0)
        eval_s = ns_to_s(data.get("eval_duration"))
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_s = ns_to_s(data.get("prompt_eval_duration"))
        tok_per_s = eval_count / eval_s if eval_s > 0 else 0
        prompt_tok_per_s = prompt_eval_count / prompt_eval_s if prompt_eval_s > 0 else 0
        tok_rates.append(tok_per_s)
        print(f"  run {i+1}: wall={wall_s:.2f}s  gen={eval_count}tok/{eval_s:.2f}s="
              f"{tok_per_s:.1f} tok/s  |  prompt={prompt_eval_count}tok/{prompt_eval_s:.3f}s="
              f"{prompt_tok_per_s:.1f} tok/s")

    avg = sum(tok_rates) / len(tok_rates)
    print(f"  --> avg generation speed: {avg:.1f} tok/s")
