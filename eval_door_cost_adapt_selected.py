# eval_door_cost_adapt_selected.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from eval_door_transfer import ChainSpec, ModuleSpec, evaluate_chain, write_rows


TARGET = "BabyAI-UnlockPickup-v0"
BUILD_DIR = Path("runs_cost_door")
ADAPT_DIR = Path("runs_cost_adapt_door")

CHECKPOINTS = [
    ("A0000", 0),
    ("A0175", 17500),
    ("A035", 35000),
    ("A0525", 52500),
    ("A070", 70000),
    ("A105", 105000),
    ("A140", 140000),
    ("A210", 210000),
    ("A280", 280000),
]


def p(*parts) -> str:
    return str(Path(*parts))


def b_cost_split(L: int):
    base = int(round(5 * L / 7))
    agg = L - base
    return base, agg


def b_g1_l420_chain(tag: str, c_adapt: int) -> ChainSpec:
    L = 420000
    base, agg = b_cost_split(L)

    if c_adapt == 0:
        near_key = p(BUILD_DIR, "B_L420_G1_near_key_step2_unlockpickup")
        nk_hk = p(BUILD_DIR, "B_L420_G1_near_key_to_has_key_step2_unlockpickup")
        hk_od = p(BUILD_DIR, "B_L420_G1_has_key_to_opened_door_step2_unlockpickup")
    else:
        near_key = p(ADAPT_DIR, f"B_L420_{tag}_G1_near_key")
        nk_hk = p(ADAPT_DIR, f"B_L420_{tag}_G1_near_key_to_has_key")
        hk_od = p(ADAPT_DIR, f"B_L420_{tag}_G1_has_key_to_opened_door")

    return ChainSpec(
        name=f"B_seq_G1_L420_{tag}",
        method="B_sequential_cost_adapt",
        source_label="DoorKey->UnlockLocal->UnlockPickup",
        granularity="G1",
        env_id=TARGET,
        final_target="opened_door",
        c_single=base,
        c_agg=agg,
        c_adapt=c_adapt,
        modules=[
            ModuleSpec("near_key", near_key, "near_key"),
            ModuleSpec("near_key_to_has_key", nk_hk, "has_key"),
            ModuleSpec("has_key_to_opened_door", hk_od, "opened_door"),
        ],
    )


def b_g2_l245_chain(tag: str, c_adapt: int) -> ChainSpec:
    L = 245000
    base, agg = b_cost_split(L)

    if c_adapt == 0:
        approach_key = p(BUILD_DIR, "B_L245_G2_approach_key_step2_unlockpickup")
        pickup_key = p(BUILD_DIR, "B_L245_G2_pickup_key_step2_unlockpickup")
        approach_door = p(BUILD_DIR, "B_L245_G2_approach_door_step2_unlockpickup")
        toggle_door = p(BUILD_DIR, "B_L245_G2_toggle_door_step2_unlockpickup")
    else:
        approach_key = p(ADAPT_DIR, f"B_L245_{tag}_G2_approach_key")
        pickup_key = p(ADAPT_DIR, f"B_L245_{tag}_G2_pickup_key")
        approach_door = p(ADAPT_DIR, f"B_L245_{tag}_G2_approach_door")
        toggle_door = p(ADAPT_DIR, f"B_L245_{tag}_G2_toggle_door")

    return ChainSpec(
        name=f"B_seq_G2_L245_{tag}",
        method="B_sequential_cost_adapt",
        source_label="DoorKey->UnlockLocal->UnlockPickup",
        granularity="G2",
        env_id=TARGET,
        final_target="opened_door",
        c_single=base,
        c_agg=agg,
        c_adapt=c_adapt,
        modules=[
            ModuleSpec("approach_key", approach_key, "approach_key", external_check="near_key"),
            ModuleSpec("pickup_key", pickup_key, "pickup_key", external_check="has_key"),
            ModuleSpec("approach_door", approach_door, "approach_door", external_check="door_ready"),
            ModuleSpec("toggle_door", toggle_door, "toggle_door", external_check="opened_door"),
        ],
    )


def build_specs() -> List[ChainSpec]:
    specs: List[ChainSpec] = []
    for tag, c_adapt in CHECKPOINTS:
        specs.append(b_g1_l420_chain(tag, c_adapt))
        specs.append(b_g2_l245_chain(tag, c_adapt))
    return specs


def summarize_convergence(rows: List[Dict], threshold: float = 0.95, plateau_delta: float = 0.02) -> List[Dict]:
    groups: Dict[str, List[Dict]] = {}

    for row in rows:
        parts = row["chain_name"].split("_")
        # B_seq_G1_L420_A0175 -> B_seq_G1_L420
        group = "_".join(parts[:4])
        groups.setdefault(group, []).append(row)

    summary = []
    for group, items in groups.items():
        items = sorted(items, key=lambda r: int(r["c_adapt"]))
        selected = items[-1]
        reason = "reached_E2"

        for i, row in enumerate(items):
            s = float(row["final_success_rate"])
            if s >= threshold:
                selected = row
                reason = f"success_ge_{threshold:.2f}"
                break

            if i >= 2:
                s0 = float(items[i - 2]["final_success_rate"])
                s1 = float(items[i - 1]["final_success_rate"])
                s2 = float(items[i]["final_success_rate"])
                if (s2 - s1) < plateau_delta and (s1 - s0) < plateau_delta:
                    selected = row
                    reason = f"plateau_two_gains_lt_{plateau_delta:.2f}"
                    break

        zero = items[0]
        summary.append({
            "config": group,
            "method": selected["method"],
            "granularity": selected["granularity"],
            "c_build": int(selected["c_build"]),
            "zero_shot_success": float(zero["final_success_rate"]),
            "zero_shot_avg_steps": float(zero["avg_total_steps"]),
            "c_adapt_at_conv": int(selected["c_adapt"]),
            "c_total_at_conv": int(selected["c_total"]),
            "converged_success": float(selected["final_success_rate"]),
            "converged_avg_steps": float(selected["avg_total_steps"]),
            "conv_reason": reason,
        })

    return sorted(summary, key=lambda r: (r["granularity"], r["c_build"]))


def write_simple_csv(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_cost_adapt_door/door_cost_adapt_selected_curve_ep300.csv")
    ap.add_argument("--summary_csv", type=str, default="results_cost_adapt_door/door_cost_adapt_selected_summary_ep300.csv")
    ap.add_argument("--only_granularity", type=str, default="", help="Optional: G1 or G2")
    args = ap.parse_args()

    specs = build_specs()
    if args.only_granularity:
        specs = [s for s in specs if s.granularity == args.only_granularity]

    rows = []
    for spec in specs:
        print(f"\n[EVAL] {spec.name} ({spec.granularity}, C_build={spec.c_single + spec.c_agg}, C_adapt={spec.c_adapt})")
        row = evaluate_chain(spec, episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        print({
            k: row[k]
            for k in [
                "chain_name",
                "granularity",
                "c_build",
                "c_adapt",
                "c_total",
                "final_success_rate",
                "avg_total_steps",
            ]
        })
        rows.append(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved curve: {out_csv}")

    summary = summarize_convergence(rows)
    summary_csv = Path(args.summary_csv)
    write_simple_csv(summary, summary_csv)
    print(f"Saved summary: {summary_csv}")

    print("\nConvergence summary:")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
