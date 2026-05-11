# eval_door_cost_build_scan.py
# Evaluate B/C x G1/G2 C_build scan zero-shot transfer to BabyAI-UnlockPickup-v0.
#
# Put this file in:
#   D:\Project\HRC_granularity\eval_door_cost_build_scan.py
#
# Requires:
#   eval_door_transfer.py

from __future__ import annotations

import argparse
from pathlib import Path

from eval_door_transfer import ChainSpec, ModuleSpec, evaluate_chain, write_rows


TARGET = "BabyAI-UnlockPickup-v0"

RC = Path("runs_cost_door")
RT = Path("runs_transfer_door")
RJ = Path("runs_joint_door")


def p(*parts) -> str:
    return str(Path(*parts))


def b_cost_split(L: int):
    # B sequential co-scaled build:
    # DoorKey base = 5L/7, source aggregation = 2L/7
    base = int(round(5 * L / 7))
    agg = L - base
    return base, agg


def b_g1_chain(L: int) -> ChainSpec:
    if L == 490_000:
        near_key = p(RT, "B_SEQ_G1_near_key_step2_unlockpickup")
        nk_hk = p(RT, "B_SEQ_G1_near_key_to_has_key_step2_unlockpickup")
        hk_od = p(RT, "B_SEQ_G1_has_key_to_opened_door_step2_unlockpickup")
    else:
        tag = L // 1000
        near_key = p(RC, f"B_L{tag}_G1_near_key_step2_unlockpickup")
        nk_hk = p(RC, f"B_L{tag}_G1_near_key_to_has_key_step2_unlockpickup")
        hk_od = p(RC, f"B_L{tag}_G1_has_key_to_opened_door_step2_unlockpickup")

    base, agg = b_cost_split(L)
    return ChainSpec(
        name=f"B_seq_G1_L{L//1000}_adapt0",
        method="B_sequential_cost_scan",
        source_label="DoorKey->UnlockLocal->UnlockPickup",
        granularity="G1",
        env_id=TARGET,
        final_target="opened_door",
        c_single=base,
        c_agg=agg,
        c_adapt=0,
        modules=[
            ModuleSpec("near_key", near_key, "near_key"),
            ModuleSpec("near_key_to_has_key", nk_hk, "has_key"),
            ModuleSpec("has_key_to_opened_door", hk_od, "opened_door"),
        ],
    )


def b_g2_chain(L: int) -> ChainSpec:
    if L == 490_000:
        approach_key = p(RT, "B_SEQ_G2_approach_key_step2_unlockpickup")
        pickup_key = p(RT, "B_SEQ_G2_pickup_key_step2_unlockpickup")
        approach_door = p(RT, "B_SEQ_G2_approach_door_step2_unlockpickup")
        toggle_door = p(RT, "B_SEQ_G2_toggle_door_step2_unlockpickup")
    else:
        tag = L // 1000
        approach_key = p(RC, f"B_L{tag}_G2_approach_key_step2_unlockpickup")
        pickup_key = p(RC, f"B_L{tag}_G2_pickup_key_step2_unlockpickup")
        approach_door = p(RC, f"B_L{tag}_G2_approach_door_step2_unlockpickup")
        toggle_door = p(RC, f"B_L{tag}_G2_toggle_door_step2_unlockpickup")

    base, agg = b_cost_split(L)
    return ChainSpec(
        name=f"B_seq_G2_L{L//1000}_adapt0",
        method="B_sequential_cost_scan",
        source_label="DoorKey->UnlockLocal->UnlockPickup",
        granularity="G2",
        env_id=TARGET,
        final_target="opened_door",
        c_single=base,
        c_agg=agg,
        c_adapt=0,
        modules=[
            ModuleSpec("approach_key", approach_key, "approach_key", external_check="near_key"),
            ModuleSpec("pickup_key", pickup_key, "pickup_key", external_check="has_key"),
            ModuleSpec("approach_door", approach_door, "approach_door", external_check="door_ready"),
            ModuleSpec("toggle_door", toggle_door, "toggle_door", external_check="opened_door"),
        ],
    )


def c_g1_chain(L: int) -> ChainSpec:
    if L == 490_000:
        near_key = p(RJ, "C_JOINT_G1_near_key")
        nk_hk = p(RJ, "C_JOINT_G1_near_key_to_has_key")
        hk_od = p(RJ, "C_JOINT_G1_has_key_to_opened_door")
    else:
        tag = L // 1000
        near_key = p(RC, f"C_L{tag}_G1_near_key")
        nk_hk = p(RC, f"C_L{tag}_G1_near_key_to_has_key")
        hk_od = p(RC, f"C_L{tag}_G1_has_key_to_opened_door")

    return ChainSpec(
        name=f"C_joint_G1_L{L//1000}_adapt0",
        method="C_joint_cost_scan",
        source_label="DoorKey+UnlockLocal+UnlockPickup",
        granularity="G1",
        env_id=TARGET,
        final_target="opened_door",
        c_single=0,
        c_agg=L,
        c_adapt=0,
        modules=[
            ModuleSpec("near_key", near_key, "near_key"),
            ModuleSpec("near_key_to_has_key", nk_hk, "has_key"),
            ModuleSpec("has_key_to_opened_door", hk_od, "opened_door"),
        ],
    )


def c_g2_chain(L: int) -> ChainSpec:
    if L == 490_000:
        approach_key = p(RJ, "C_JOINT_G2_approach_key")
        pickup_key = p(RJ, "C_JOINT_G2_pickup_key")
        approach_door = p(RJ, "C_JOINT_G2_approach_door")
        toggle_door = p(RJ, "C_JOINT_G2_toggle_door")
    else:
        tag = L // 1000
        approach_key = p(RC, f"C_L{tag}_G2_approach_key")
        pickup_key = p(RC, f"C_L{tag}_G2_pickup_key")
        approach_door = p(RC, f"C_L{tag}_G2_approach_door")
        toggle_door = p(RC, f"C_L{tag}_G2_toggle_door")

    return ChainSpec(
        name=f"C_joint_G2_L{L//1000}_adapt0",
        method="C_joint_cost_scan",
        source_label="DoorKey+UnlockLocal+UnlockPickup",
        granularity="G2",
        env_id=TARGET,
        final_target="opened_door",
        c_single=0,
        c_agg=L,
        c_adapt=0,
        modules=[
            ModuleSpec("approach_key", approach_key, "approach_key", external_check="near_key"),
            ModuleSpec("pickup_key", pickup_key, "pickup_key", external_check="has_key"),
            ModuleSpec("approach_door", approach_door, "approach_door", external_check="door_ready"),
            ModuleSpec("toggle_door", toggle_door, "toggle_door", external_check="opened_door"),
        ],
    )


def build_specs():
    budgets = [245_000, 350_000, 420_000, 490_000]
    specs = []
    for L in budgets:
        specs.append(b_g1_chain(L))
        specs.append(b_g2_chain(L))
        specs.append(c_g1_chain(L))
        specs.append(c_g2_chain(L))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_cost_door/door_cost_build_scan_ep300.csv")
    ap.add_argument("--only_method", type=str, default="", help="Optional: B_sequential_cost_scan or C_joint_cost_scan")
    ap.add_argument("--only_granularity", type=str, default="", help="Optional: G1 or G2")
    ap.add_argument("--only_budget", type=int, default=0, help="Optional C_build budget, e.g. 245000")
    args = ap.parse_args()

    specs = build_specs()
    if args.only_method:
        specs = [s for s in specs if s.method == args.only_method]
    if args.only_granularity:
        specs = [s for s in specs if s.granularity == args.only_granularity]
    if args.only_budget:
        specs = [s for s in specs if (s.c_single + s.c_agg) == args.only_budget]

    rows = []
    for spec in specs:
        print(f"\n[EVAL] {spec.name} ({spec.method}, {spec.granularity}, C_build={spec.c_single + spec.c_agg})")
        row = evaluate_chain(spec, episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        print({
            k: row[k]
            for k in [
                "chain_name",
                "method",
                "granularity",
                "c_build",
                "c_total",
                "final_success_rate",
                "avg_total_steps",
            ]
        })
        rows.append(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
