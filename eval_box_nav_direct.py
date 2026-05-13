from pathlib import Path
import argparse
import pandas as pd

from eval_chain_g2 import (
    G2ChainSpec,
    G2ModuleSpec,
    evaluate_g2_chain,
    write_rows,
)

def model(run_name: str, filename: str) -> str:
    return str(Path("runs_box_nav") / run_name / filename)

def build_specs():
    return [
        G2ChainSpec(
            name="nav_empty8x8_goal_g2",
            env_id="MiniGrid-Empty-8x8-v0",
            final_target="at_goal",
            total_train_steps=150_000,
            modules=[
                G2ModuleSpec(
                    name="approach_goal",
                    model_path=model(
                        "NAV_empty8x8_approach_goal_s0",
                        "MiniGrid-Empty-8x8-v0_final.zip",
                    ),
                    template="approach_goal",
                    external_check="at_goal",
                    horizon=128,
                ),
            ],
        ),

        G2ChainSpec(
            name="nav_multiroom_n2s4_goal_g2",
            env_id="MiniGrid-MultiRoom-N2-S4-v0",
            final_target="at_goal",
            total_train_steps=300_000,
            modules=[
                G2ModuleSpec(
                    name="approach_goal",
                    model_path=model(
                        "NAV_multiroom_n2s4_approach_goal_s0",
                        "MiniGrid-MultiRoom-N2-S4-v0_final.zip",
                    ),
                    template="approach_goal",
                    external_check="at_goal",
                    horizon=256,
                ),
            ],
        ),

        # KeyInBox box chain:
        # approach_box -> toggle_box
        #
        # Note:
        # external_check="door_ready" is only used here to trigger
        # eval_chain_g2.py's internal info["is_success"] branch.
        # It does NOT mean approach_box is checking a door.
        G2ChainSpec(
            name="box_keyinbox_open_box_chain_g2",
            env_id="BabyAI-KeyInBox-v0",
            final_target="opened_box",
            total_train_steps=600_000,
            modules=[
                G2ModuleSpec(
                    name="approach_box",
                    model_path=model(
                        "BOX_keyinbox_approach_box_s0",
                        "BabyAI-KeyInBox-v0_final.zip",
                    ),
                    template="approach_box",
                    external_check="door_ready",
                    horizon=128,
                ),
                G2ModuleSpec(
                    name="toggle_box",
                    model_path=model(
                        "BOX_keyinbox_toggle_box_s0",
                        "BabyAI-KeyInBox-v0_final.zip",
                    ),
                    template="toggle_box",
                    external_check="opened_box",
                    horizon=128,
                ),
            ],
        ),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="results_box_nav/box_nav_direct_eval_ep300.csv")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    rows = []
    for spec in build_specs():
        print(f"\n[EVAL] {spec.name}")
        row = evaluate_g2_chain(
            spec,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=(not args.stochastic),
        )
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved: {out_csv}")

    df = pd.read_csv(out_csv)
    keep = [
        "chain_name",
        "env_id",
        "final_success_rate",
        "chain_completion_rate",
        "avg_total_steps",
    ]
    module_cols = [
        c for c in df.columns
        if c.startswith("module_success/") or c.startswith("module_steps/")
    ]
    print("\nSummary:")
    print(df[keep + module_cols].T)

if __name__ == "__main__":
    main()
