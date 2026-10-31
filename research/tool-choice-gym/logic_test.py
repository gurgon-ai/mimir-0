#!/usr/bin/env python
"""Five logic problems, both models, full transcripts."""
import requests

BASE_URL = "http://127.0.0.1:11434"
MODELS = ["qwen3.8:27b", "gemma4:26b"]

PROBLEMS = [
    ("bat_and_ball",
     "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
     "ball. How much does the ball cost? Show your reasoning."),
    ("widgets",
     "If it takes 5 machines 5 minutes to make 5 widgets, how long would it "
     "take 100 machines to make 100 widgets? Show your reasoning."),
    ("syllogism",
     "Consider this argument: 'All roses are flowers. Some flowers fade "
     "quickly. Therefore, some roses fade quickly.' Is this argument "
     "logically valid? Explain why or why not."),
    ("riddle",
     "A man looks at a photograph and says: 'Brothers and sisters I have "
     "none, but that man's father is my father's son.' Who is in the "
     "photograph? Explain your reasoning."),
    ("guards",
     "You face two doors, one leading to safety and one to danger. Two "
     "guards stand there; one always tells the truth, one always lies, but "
     "you don't know which is which. You may ask exactly one guard exactly "
     "one question. What question do you ask to determine the safe door, "
     "and why does it work?"),
]

for qid, prompt in PROBLEMS:
    print(f"\n\n{'='*70}\nQUESTION [{qid}]: {prompt}\n{'='*70}")
    for model in MODELS:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 3000, "temperature": 0.7},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        text = (msg.get("content") or "").strip()
        thinking = (msg.get("thinking") or "").strip()
        eval_count = data.get("eval_count", 0)
        done_reason = data.get("done_reason", "")
        print(f"\n--- {model} (thinking={len(thinking)} chars, "
              f"eval_count={eval_count}, done_reason={done_reason}) ---\n{text}\n")
