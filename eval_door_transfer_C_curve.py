# eval_door_transfer_C_curve.py
# Evaluate C Joint Training adaptation curve on BabyAI-UnlockPickup-v0 opened_door.
# Put this file in: D:\Project\HRC_granularity\eval_door_transfer_C_curve.py
#
# Requires eval_door_transfer.py in the same project root.

from __future__ import annotations

import argparse
from pathlib import Path

from eval_door_transfer import ChainSpec, ModuleSpec, evaluate_chain, write_rows


TARGET = "BabyAI-UnlockPickup-v0"
RJ = Path("runs_joint_door")


def p(*parts) -> str:
    return str(Path(*parts))


def g0_chain(name: str, has_key_dir: str, opened_door_dir: str, c_adapt: int) -> ChainSpec:
    return ChainSpec(
        name=name,
        method="C_joint",
        source_label="DoorKey+UnlockLocal+UnlockPickup",
        granularity="G0",
        env_id=TARGET,
        final_target="opened_door",
        c_single=0,
        c_agg=490_000,
        c_adapt=c_adapt,
        modules=[
            ModuleSpec("has_key", has_key_dir, "has_key"),
            ModuleSpec("opened_door", opened_door_dir, "opened_door"),
        ],
    )


def g1_chain(name: str, near_key_dir: str, nk_hk_dir: str, hk_od_dir: str, c_adapt: int) -> ChainSpec:
    return ChainSpec(
        name=name,
        method="C_joint",
        source_label="DoorKey+UnlockLocal+UnlockPickup",
        granularity="G1",
        env_id=TARGET,
        final_target="opened_door",
        c_single=0,
        c_agg=490_000,
        c_adapt=c_adapt,
        modules=[
            ModuleSpec("near_key", near_key_dir, "near_key"),
            ModuleSpec("near_key_to_has_key", nk_hk_dir, "has_key"),
            ModuleSpec("has_key_to_opened_door", hk_od_dir, "opened_door"),
        ],
    )


def g2_chain(
    name: str,
    approach_key_dir: str,
    pickup_key_dir: str,
    approach_door_dir: str,
    toggle_door_dir: str,
    c_adapt: int,
) -> ChainSpec:
    return ChainSpec(
        name=name,
        method="C_joint",
        source_label="DoorKey+UnlockLocal+UnlockPickup",
        granularity="G2",
        env_id=TARGET,
        final_target="opened_door",
        c_single=0,
        c_agg=490_000,
        c_adapt=c_adapt,
        modules=[
            ModuleSpec("approach_key", approach_key_dir, "approach_key", external_check="near_key"),
            ModuleSpec("pickup_key", pickup_key_dir, "pickup_key", external_check="has_key"),
            ModuleSpec("approach_door", approach_door_dir, "approach_door", external_check="door_ready"),
            ModuleSpec("toggle_door", toggle_door_dir, "toggle_door", external_check="opened_door"),
        ],
    )


def build_specs():
    specs = []

    # adapt0: source joint model directly transferred to target
    specs += [
        g0_chain("C_joint_G0_adapt0", p(RJ, "C_JOINT_G0_has_key"), p(RJ, "C_JOINT_G0_opened_door"), 0),
        g1_chain(
            "C_joint_G1_adapt0",
            p(RJ, "C_JOINT_G1_near_key"),
            p(RJ, "C_JOINT_G1_near_key_to_has_key"),
            p(RJ, "C_JOINT_G1_has_key_to_opened_door"),
            0,
        ),
        g2_chain(
            "C_joint_G2_adapt0",
            p(RJ, "C_JOINT_G2_approach_key"),
            p(RJ, "C_JOINT_G2_pickup_key"),
            p(RJ, "C_JOINT_G2_approach_door"),
            p(RJ, "C_JOINT_G2_toggle_door"),
            0,
        ),
    ]

    # adapt35/70/140: target-adapted models
    for budget in [35_000, 70_000, 140_000]:
        tag = budget // 1000
        specs += [
            g0_chain(
                f"C_joint_G0_adapt{tag}",
                p(RJ, f"C_ADAPT{tag}_G0_has_key"),
                p(RJ, f"C_ADAPT{tag}_G0_opened_door"),
                budget,
            ),
            g1_chain(
                f"C_joint_G1_adapt{tag}",
                p(RJ, f"C_ADAPT{tag}_G1_near_key"),
                p(RJ, f"C_ADAPT{tag}_G1_near_key_to_has_key"),
                p(RJ, f"C_ADAPT{tag}_G1_has_key_to_opened_door"),
                budget,
            ),
            g2_chain(
                f"C_joint_G2_adapt{tag}",
                p(RJ, f"C_ADAPT{tag}_G2_approach_key"),
                p(RJ, f"C_ADAPT{tag}_G2_pickup_key"),
                p(RJ, f"C_ADAPT{tag}_G2_approach_door"),
                p(RJ, f"C_ADAPT{tag}_G2_toggle_door"),
                budget,
            ),
        ]

    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_joint_door/door_transfer_C_joint_curve_ep300.csv")
    ap.add_argument("--only_granularity", type=str, default="", help="Optional: G0, G1, or G2")
    args = ap.parse_args()

    specs = build_specs()
    if args.only_granularity:
        specs = [s for s in specs if s.granularity == args.only_granularity]

    rows = []
    for spec in specs:
        print(f"\n[EVAL] {spec.name} ({spec.granularity}, c_adapt={spec.c_adapt})")
        row = evaluate_chain(spec, episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        print({k: row[k] for k in ["chain_name", "granularity", "c_total", "final_success_rate", "avg_total_steps"]})
        rows.append(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
