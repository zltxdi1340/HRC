cd D:\Project\HRC_granularity

mkdir runs_box_nav -Force
mkdir results_box_nav -Force
mkdir results_box_nav\box -Force
mkdir results_box_nav\nav -Force

python -m rl.train_sb3_g2_ppo `
  --env_id MiniGrid-Empty-8x8-v0 `
  --template approach_goal `
  --total_timesteps 150000 `
  --n_envs 8 `
  --seed 0 `
  --option_horizon 128 `
  --logdir runs_box_nav `
  --run_name NAV_empty8x8_approach_goal_s0

python -m rl.train_sb3_g2_ppo `
  --env_id MiniGrid-FourRooms-v0 `
  --template approach_goal `
  --total_timesteps 300000 `
  --n_envs 8 `
  --seed 0 `
  --option_horizon 256 `
  --logdir runs_box_nav `
  --run_name NAV_fourrooms_approach_goal_s0

python -m rl.train_sb3_g2_ppo `
  --env_id MiniGrid-MultiRoom-N2-S4-v0 `
  --template approach_goal `
  --total_timesteps 300000 `
  --n_envs 8 `
  --seed 0 `
  --option_horizon 256 `
  --logdir runs_box_nav `
  --run_name NAV_multiroom_n2s4_approach_goal_s0

python -m rl.train_sb3_g2_ppo `
  --env_id BabyAI-KeyInBox-v0 `
  --template approach_box `
  --total_timesteps 300000 `
  --n_envs 8 `
  --seed 0 `
  --option_horizon 128 `
  --logdir runs_box_nav `
  --run_name BOX_keyinbox_approach_box_s0

python -m rl.train_sb3_g2_ppo `
  --env_id BabyAI-KeyInBox-v0 `
  --template toggle_box `
  --parent_goal near_target `
  --total_timesteps 300000 `
  --n_envs 8 `
  --seed 0 `
  --option_horizon 128 `
  --logdir runs_box_nav `
  --run_name BOX_keyinbox_toggle_box_s0