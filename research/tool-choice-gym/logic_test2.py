#!/usr/bin/env python
"""Round 2: altered-number variants + one novel puzzle, to control for memorization."""
import requests

BASE_URL = "http://127.0.0.1:11434"
MODELS = ["qwen3.8:27b", "gemma4:26b"]

PROBLEMS = [
    ("wrench_hammer",
     "A wrench and a hammer together cost $47.30. The hammer costs $37.00 "
     "more than the wrench. How much does the wrench cost? Show your "
     "reasoning."),
    ("robots_toys",
     "If it takes 7 robots 9 minutes to assemble 7 toys, how long would it "
     "take 21 robots to assemble 21 toys? Show your reasoning."),
    ("valid_syllogism",
     "Consider this argument: 'All engineers at this company have a "
     "degree. All employees with a degree get a signing bonus. Therefore, "
     "all engineers at this company get a signing bonus.' Is this argument "
     "logically valid? Explain why or why not."),
    ("daughter_riddle",
     "A woman points to a picture and says: 'Her mother is my mother's "
     "only daughter, and I have no sisters.' Who is in the picture? "
     "Explain your reasoning."),
    ("seating_puzzle",
     "Five friends - Ana, Ben, Cara, Dan, and Eve - sit in a row of five "
     "chairs numbered 1 to 5, left to right. Clues: (1) Ana sits somewhere "
     "to the left of Ben. (2) Cara sits immediately next to Dan (either "
     "side). (3) Eve sits in chair 3. (4) Ben does not sit in chair 5. "
     "(5) Dan sits in an odd-numbered chair. Determine exactly who sits in "
     "each chair, and show your reasoning."),
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
