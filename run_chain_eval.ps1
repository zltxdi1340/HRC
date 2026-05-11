cd D:\Project\HRC_granularity

mkdir results_granularity -ErrorAction SilentlyContinue

# Full fair-budget chain evaluation.
python eval_chain_granularity.py --preset all --episodes 100 --seed 0 --out_csv results_granularity\chain_eval.csv

# Optional deterministic evaluation, useful as a robustness check.
# python eval_chain_granularity.py --preset all --episodes 100 --seed 0 --deterministic --out_csv results_granularity\chain_eval_deterministic.csv
