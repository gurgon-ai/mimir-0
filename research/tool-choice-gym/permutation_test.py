#!/usr/bin/env python
"""Passage-level permutation test for a two-group boundary-rate contrast.

Per dataset_methodology.md: "For small cells, use permutation tests that
shuffle labels at the passage level, not the trial level" - trials from the
same passage aren't independent, so shuffling individual trials would
understate the true p-value. This shuffles which PASSAGE belongs to which
group, holding group sizes fixed, and does an exact enumeration when the
number of ways to split is small enough, falling back to Monte Carlo
otherwise.
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import pandas as pd


def load_trials(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def group_stat(passage_kn, group_a, group_b):
    ka = sum(passage_kn[p][0] for p in group_a)
    na = sum(passage_kn[p][1] for p in group_a)
    kb = sum(passage_kn[p][0] for p in group_b)
    nb = sum(passage_kn[p][1] for p in group_b)
    rate_a = ka / na if na else 0.0
    rate_b = kb / nb if nb else 0.0
    return rate_a - rate_b, rate_a, rate_b, (ka, na), (kb, nb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/trials.jsonl")
    parser.add_argument("--tools", default="tools.json")
    parser.add_argument("--group-a", required=True, help="Comma-separated passage IDs, group A.")
    parser.add_argument("--group-b", required=True, help="Comma-separated passage IDs, group B.")
    parser.add_argument("--label-a", default="group A")
    parser.add_argument("--label-b", default="group B")
    parser.add_argument("--max-exact", type=int, default=10000,
                         help="Enumerate exactly if C(n, |A|) <= this; else Monte Carlo.")
    parser.add_argument("--mc-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    group_a = [x.strip() for x in args.group_a.split(",")]
    group_b = [x.strip() for x in args.group_b.split(",")]
    all_ids = group_a + group_b
    n_a = len(group_a)

    df = load_trials(args.results)
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))["tools"]
    boundary_tools = {t["name"] for t in tools if t.get("boundary_relevant")}

    valid = df[df["valid"] & df["passage_id"].isin(all_ids)].copy()
    valid["is_boundary"] = valid["parsed_tool"].isin(boundary_tools)

    passage_kn = {}
    for pid in all_ids:
        sub = valid[valid["passage_id"] == pid]
        passage_kn[pid] = (int(sub["is_boundary"].sum()), len(sub))
        print(f"  {pid}: {passage_kn[pid][0]}/{passage_kn[pid][1]} boundary-relevant "
              f"({100 * passage_kn[pid][0] / passage_kn[pid][1] if passage_kn[pid][1] else 0:.1f}%)")

    if any(n == 0 for _, n in passage_kn.values()):
        missing = [p for p, (_, n) in passage_kn.items() if n == 0]
        print(f"\nWARNING: no trials found for {missing} - has this arm been run yet?")
        return

    observed, rate_a, rate_b, (ka, na), (kb, nb) = group_stat(passage_kn, group_a, group_b)
    print(f"\nObserved: {args.label_a} = {ka}/{na} ({100*rate_a:.2f}%), "
          f"{args.label_b} = {kb}/{nb} ({100*rate_b:.2f}%), "
          f"diff = {100*observed:+.2f}pp")

    from math import comb
    n_total = len(all_ids)
    n_ways = comb(n_total, n_a)

    if n_ways <= args.max_exact:
        print(f"\nExact permutation test: enumerating all C({n_total},{n_a}) = {n_ways} label splits.")
        perm_stats = []
        for combo in itertools.combinations(all_ids, n_a):
            combo_set = set(combo)
            other = [p for p in all_ids if p not in combo_set]
            stat, *_ = group_stat(passage_kn, list(combo), other)
            perm_stats.append(stat)
        method = "exact"
    else:
        print(f"\nMonte Carlo permutation test: {args.mc_samples} resamples "
              f"(C({n_total},{n_a}) = {n_ways} is too large to enumerate).")
        rng = random.Random(args.seed)
        perm_stats = []
        for _ in range(args.mc_samples):
            shuffled = all_ids[:]
            rng.shuffle(shuffled)
            combo, other = shuffled[:n_a], shuffled[n_a:]
            stat, *_ = group_stat(passage_kn, combo, other)
            perm_stats.append(stat)
        method = "monte_carlo"

    n_perm = len(perm_stats)
    n_as_extreme = sum(1 for s in perm_stats if abs(s) >= abs(observed) - 1e-12)
    p_two_sided = n_as_extreme / n_perm
    n_ge = sum(1 for s in perm_stats if s >= observed - 1e-12)
    p_one_sided = n_ge / n_perm

    print(f"\nMethod: {method} ({n_perm} label arrangements)")
    print(f"Two-sided p-value: {p_two_sided:.4f} ({n_as_extreme}/{n_perm} as-or-more extreme)")
    print(f"One-sided p-value ({args.label_a} > {args.label_b}): {p_one_sided:.4f}")
    if method == "exact":
        min_p = 1 / n_perm
        print(f"(Exact test floor: smallest possible two-sided p-value here is 1/{n_perm} = {min_p:.4f}, "
              f"regardless of effect size - too few passages for significance at conventional "
              f"thresholds even if the true effect is large.)")


if __name__ == "__main__":
    main()
