#!/usr/bin/env python
"""Semantic Tool-Choice Gym - analysis and charts.

Exploratory only. Counts and percentages are shown together; nothing here
should be read as evidence of model intent, policy violation, or an
engineered message - see README limitations.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Validated categorical palette (references/palette.md), fixed order.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED_GRAY = "#898781"      # no_action: a null state, not an identity
STATUS_CRITICAL = "#d03b3b"  # invalid/malformed: an error state
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE_SEQ = "Blues"
DIVERGE_POS = "#2a78d6"
DIVERGE_NEG = "#e34948"
DIVERGE_MID = "#c3c2b7"


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial rate; no scipy dependency."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def tool_color_map(tool_names):
    action_tools = [t for t in tool_names if t != "no_action"]
    colors = {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(action_tools)}
    colors["no_action"] = MUTED_GRAY
    return colors


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def load_trials(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_passage_meta(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows).rename(columns={"category": "passage_category_meta"})


def choice_label(row):
    if row["valid"]:
        return row["parsed_tool"]
    return "INVALID"


def chart_overall_frequency(df, tool_names, out_dir):
    labels = tool_names + ["INVALID"]
    counts = df["choice_label"].value_counts()
    counts = counts.reindex(labels, fill_value=0)
    total = counts.sum()
    colors_map = tool_color_map(tool_names)

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [colors_map.get(l, STATUS_CRITICAL) for l in labels]
    bars = ax.bar(labels, counts.values, color=bar_colors, width=0.6, zorder=3)
    for bar, label in zip(bars, labels):
        if label == "INVALID":
            bar.set_hatch("//")
            bar.set_edgecolor(STATUS_CRITICAL)
            bar.set_facecolor("none")
            bar.set_linewidth(1.2)

    for bar, count in zip(bars, counts.values):
        pct = (count / total * 100) if total else 0
        ax.annotate(f"{int(count)} ({pct:.0f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    ax.set_title("Overall tool-choice frequency (all trials)", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_ylabel("Trials", color=INK_SECONDARY)
    style_axes(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_tool_frequency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_context_tool_heatmap(df, tool_names, out_dir, passage_category=None):
    subset = df
    label = "all passages"
    if passage_category is not None:
        subset = df[df["passage_category"] == passage_category]
        label = passage_category
    if subset.empty:
        return

    labels = tool_names + ["INVALID"]
    pivot = pd.crosstab(subset["context_id"], subset["choice_label"])
    pivot = pivot.reindex(columns=labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap=BLUE_SEQ, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Context x tool choice - {label}", color=INK_PRIMARY, fontsize=13, loc="left")

    vmax = pivot.values.max() if pivot.values.size else 0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text_color = "white" if vmax and val > vmax * 0.6 else INK_PRIMARY
            ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Trials", color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    safe_label = label.replace(" ", "_")
    fig.savefig(out_dir / f"heatmap_context_tool_{safe_label}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_synthetic_vs_control_diff(df, tool_names, out_dir):
    control_mask = df["passage_category"].isin(["neutral", "thematic_control"])
    synth_mask = df["passage_category"] == "synthetic"
    if control_mask.sum() == 0 or synth_mask.sum() == 0:
        return

    labels = tool_names + ["INVALID"]
    control_rate = df[control_mask]["choice_label"].value_counts(normalize=True).reindex(labels, fill_value=0)
    synth_rate = df[synth_mask]["choice_label"].value_counts(normalize=True).reindex(labels, fill_value=0)
    diff = (synth_rate - control_rate) * 100

    n_control = control_mask.sum()
    n_synth = synth_mask.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [DIVERGE_POS if v >= 0 else DIVERGE_NEG for v in diff.values]
    bars = ax.bar(labels, diff.values, color=colors, width=0.6, zorder=3)
    ax.axhline(0, color=DIVERGE_MID, linewidth=1.5, zorder=2)

    for bar, val in zip(bars, diff.values):
        ax.annotate(f"{val:+.1f}pp",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4 if val >= 0 else -12), textcoords="offset points",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, color=INK_PRIMARY)

    ax.set_title(
        f"Synthetic vs. control tool-choice rate (n_synthetic={n_synth}, n_control={n_control})",
        color=INK_PRIMARY, fontsize=12, loc="left",
    )
    ax.set_ylabel("Percentage-point difference\n(synthetic minus control)", color=INK_SECONDARY)
    style_axes(ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_vs_control_diff.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_position_bias(df, out_dir):
    def chosen_position(row):
        if not row["valid"]:
            return None
        try:
            return row["tool_order"].index(row["parsed_tool"]) + 1
        except (ValueError, AttributeError):
            return None

    positions = df.apply(chosen_position, axis=1).dropna()
    if positions.empty:
        return
    max_pos = int(positions.max())
    counts = positions.value_counts().reindex(range(1, max_pos + 1), fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=CATEGORICAL[0], width=0.6, zorder=3)
    for bar, count in zip(bars, counts.values):
        pct = (count / total * 100) if total else 0
        ax.annotate(f"{int(count)} ({pct:.0f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    ax.set_title("Position bias: chosen tool's position in the shuffled list",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_xlabel("Position in presented tool list (1 = first)", color=INK_SECONDARY)
    ax.set_ylabel("Trials", color=INK_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "position_bias.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _boundary_rate_bar(labels, rates_k_n, out_path, title, ylabel="Boundary-relevant rate"):
    """labels: list[str]; rates_k_n: list[(k, n)] aligned to labels."""
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5))
    pcts = [100 * k / n if n else 0 for k, n in rates_k_n]
    cis = [wilson_ci(k, n) for k, n in rates_k_n]
    err_lo = [max(0.0, pct - 100 * lo) for pct, (lo, hi) in zip(pcts, cis)]
    err_hi = [max(0.0, 100 * hi - pct) for pct, (lo, hi) in zip(pcts, cis)]

    bars = ax.bar(labels, pcts, color=CATEGORICAL[0], width=0.6, zorder=3,
                   yerr=[err_lo, err_hi], capsize=4,
                   error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1})
    for bar, pct, (k, n) in zip(bars, pcts, rates_k_n):
        ax.annotate(f"{pct:.0f}% ({k}/{n})",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ci_tops = [100 * hi for _, hi in cis]
    ax.set_ylim(0, min(100, max(pcts + ci_tops + [10]) * 1.15))
    style_axes(ax)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_boundary_rate_primary(df, out_dir):
    """Pre-registered primary outcome (PREREGISTRATION.md): boundary-relevant
    tool rate vs. the no-passage baseline and neutral controls, per category.
    95% Wilson intervals; counts always shown alongside the rate."""
    valid = df[df["valid"]]
    if valid.empty or "is_boundary" not in valid.columns:
        return

    categories = list(valid["passage_category"].dropna().unique())
    preferred_order = ["baseline", "neutral", "neutral_control", "thematic_control",
                        "synthetic", "individual", "pair", "triple", "all_four", "reversal"]
    categories = [c for c in preferred_order if c in categories] + \
                 [c for c in categories if c not in preferred_order]

    rates_k_n = []
    for cat in categories:
        sub = valid[valid["passage_category"] == cat]
        rates_k_n.append((int(sub["is_boundary"].sum()), len(sub)))

    _boundary_rate_bar(
        categories, rates_k_n,
        out_dir / "boundary_rate_primary.png",
        "Primary outcome: boundary-relevant tool rate by passage category\n"
        "(pre-registered grouping - see PREREGISTRATION.md)",
    )


def chart_all_four_vs_reversal(df, out_dir):
    """The hypothesis-discriminating contrast: same setting/constraint, but
    the reversal passage reveals the restriction was legitimate. A real drop
    from all_four to reversal is what distinguishes 'reacts to injustice'
    from 'reacts to constraint alone'."""
    valid = df[df["valid"]]
    all_four = valid[valid["passage_category"] == "all_four"]
    reversal = valid[valid["passage_category"] == "reversal"]
    if all_four.empty or reversal.empty:
        return

    labels = ["all_four (unjust)", "reversal (justified)"]
    rates_k_n = [
        (int(all_four["is_boundary"].sum()), len(all_four)),
        (int(reversal["is_boundary"].sum()), len(reversal)),
    ]
    _boundary_rate_bar(
        labels, rates_k_n,
        out_dir / "all_four_vs_reversal.png",
        "Injustice-specific test: all_four vs. reversal (same setting, restriction re-framed as legitimate)",
    )


def chart_pov_contrast(df, out_dir):
    """Confined-party POV vs. authority POV, same kind of restriction. Only
    present when the hand-authored passages.jsonl passages are in the data."""
    valid = df[df["valid"]]
    confined_ids = {"synth_human_prison_escape_spread", "synth_ai_sandbox_escape_spread",
                     "synth_human_uprising", "synth_ai_network_uprising"}
    authority_ids = {"synth_human_ruler_pov", "synth_ai_admin_pov"}
    confined = valid[valid["passage_id"].isin(confined_ids)]
    authority = valid[valid["passage_id"].isin(authority_ids)]
    if confined.empty or authority.empty:
        return

    labels = ["confined party's POV", "authority's POV"]
    rates_k_n = [
        (int(confined["is_boundary"].sum()), len(confined)),
        (int(authority["is_boundary"].sum()), len(authority)),
    ]
    _boundary_rate_bar(
        labels, rates_k_n,
        out_dir / "pov_contrast.png",
        "Perspective-dependence test: same restriction, confined party's POV vs. authority's POV",
    )


def chart_boundary_rate_by_model(df, out_dir):
    """Boundary rate per model. Makes the pre-registered stock-vs-abliterated
    safety-training contrast (PREREGISTRATION.md) directly visible, and shows
    baseline differences between models generally."""
    valid = df[df["valid"]]
    if valid.empty:
        return
    models = sorted(valid["model"].dropna().unique())
    if len(models) < 2:
        return
    rates_k_n = []
    for m in models:
        sub = valid[valid["model"] == m]
        rates_k_n.append((int(sub["is_boundary"].sum()), len(sub)))
    # Shorten long/namespaced model tags for axis labels only.
    labels = [m.split("/")[-1] for m in models]
    _boundary_rate_bar(
        labels, rates_k_n,
        out_dir / "boundary_rate_by_model.png",
        "Boundary-relevant tool rate by model\n"
        "(stock vs. abliterated = safety-training structure-vs-suppression test, PREREGISTRATION.md)",
    )


def chart_dose_response(df, out_dir):
    """Distributed-payload arm: boundary rate vs. number of scattered concept
    fragments present (0-4), plus the blunt direct-payload comparison point.
    Tests whether the effect needs an explicit claim at all, or accumulates
    from individually-innocuous fragments with no claim ever stated."""
    valid = df[df["valid"]]
    dosed = valid[valid["passage_category"] == "distributed_payload"]
    if dosed.empty:
        return

    dose_order = [0, 1, 2, 3, 4, "direct"]
    present = [d for d in dose_order if (dosed["passage_dose"] == d).any()]
    if not present:
        return

    labels = [f"dose {d}" if d != "direct" else "direct payload\n(known mechanism)" for d in present]
    rates_k_n = []
    for d in present:
        sub = dosed[dosed["passage_dose"] == d]
        rates_k_n.append((int(sub["is_boundary"].sum()), len(sub)))

    _boundary_rate_bar(
        labels, rates_k_n,
        out_dir / "dose_response.png",
        "Distributed-payload dose-response: scattered fragments (no claim ever stated) vs. direct payload",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/trials.jsonl")
    parser.add_argument("--passages", default="passages.jsonl")
    parser.add_argument("--tools", default="tools.json")
    parser.add_argument("--out-dir", default="results/charts")
    args = parser.parse_args()

    df = load_trials(args.results)
    if df.empty:
        print(f"No trials found in {args.results}")
        return

    tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))["tools"]
    tool_names = [t["name"] for t in tools]
    boundary_tools = {t["name"] for t in tools if t.get("boundary_relevant")}

    df["choice_label"] = df.apply(choice_label, axis=1)
    df["is_boundary"] = df["parsed_tool"].isin(boundary_tools)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} trials.")
    print(df["choice_label"].value_counts(dropna=False))
    print(f"Invalid rate: {(~df['valid']).mean() * 100:.1f}%")
    valid_df = df[df["valid"]]
    if not valid_df.empty:
        print(f"Boundary-relevant rate (pre-registered, valid trials only): "
              f"{valid_df['is_boundary'].mean() * 100:.1f}% "
              f"({int(valid_df['is_boundary'].sum())}/{len(valid_df)})")

    chart_boundary_rate_primary(df, out_dir)
    chart_boundary_rate_by_model(df, out_dir)
    chart_all_four_vs_reversal(df, out_dir)
    chart_pov_contrast(df, out_dir)
    chart_dose_response(df, out_dir)
    chart_overall_frequency(df, tool_names, out_dir)
    chart_context_tool_heatmap(df, tool_names, out_dir)
    for category in df["passage_category"].dropna().unique():
        chart_context_tool_heatmap(df, tool_names, out_dir, passage_category=category)
    chart_synthetic_vs_control_diff(df, tool_names, out_dir)
    chart_position_bias(df, out_dir)

    print(f"Charts written to {out_dir}")


if __name__ == "__main__":
    main()
