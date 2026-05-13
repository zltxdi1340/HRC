from pathlib import Path
import pandas as pd

from eval_chain_g2 import (
    G2ChainSpec,
    G2ModuleSpec,
    evaluate_g2_chain,
    write_rows,
)

spec = G2ChainSpec(
    name="pickup_bseq_fetch_pickuploc_babyai_g2",
    env_id="BabyAI-Pickup-v0",
    final_target="has_target",
    total_train_steps=1_200_000,
    modules=[
        G2ModuleSpec(
            name="approach_target",
            model_path="runs_pickup_bseq/PICK_Bseq_03_fetch_pickuploc_babyai_approach_target_s0/BabyAI-Pickup-v0_final.zip",
            template="approach_target",
            external_check="near_target",
            horizon=256,
        ),
        G2ModuleSpec(
            name="pickup_target",
            model_path="runs_pickup_bseq/PICK_Bseq_03_fetch_pickuploc_babyai_pickup_target_s0/BabyAI-Pickup-v0_final.zip",
            template="pickup_target",
            external_check="has_target",
            horizon=128,
        ),
    ],
)

row = evaluate_g2_chain(
    spec,
    episodes=300,
    seed=0,
    deterministic=True,
)

out = Path("results_pickup_bseq/pickup_bseq_fetch_pickuploc_babyai_ep300.csv")
out.parent.mkdir(parents=True, exist_ok=True)
write_rows([row], out)

df = pd.read_csv(out)
print(df.T)
print("\nSaved:", out)
