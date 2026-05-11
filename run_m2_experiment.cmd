@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/5] M2 expert on BabyAI-Pickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-Pickup-v0 --module_name M2_Pickup --parent_goal near_obj --child_goal has_obj --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 150000 --seed 0 --no_strict_parent_fail_fast --run_name M2_E_babyaipickup_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/5] M2 expert on MiniGrid-Fetch-8x8-N3-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Fetch-8x8-N3-v0 --module_name M2_Pickup --parent_goal near_obj --child_goal has_obj --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 150000 --seed 0 --no_strict_parent_fail_fast --run_name M2_E_fetch_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/5] M2 shared sequential step1 on BabyAI-Pickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-Pickup-v0 --module_name M2_Pickup --parent_goal near_obj --child_goal has_obj --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 150000 --seed 0 --no_strict_parent_fail_fast --run_name M2_SEQ_step1_babyaipickup_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/5] M2 shared sequential step2 on MiniGrid-Fetch-8x8-N3-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Fetch-8x8-N3-v0 --module_name M2_Pickup --parent_goal near_obj --child_goal has_obj --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 150000 --seed 0 --no_strict_parent_fail_fast --load_model runs_gcppo\M2_SEQ_step1_babyaipickup_nff_s0\BabyAI-Pickup-v0_final.zip --run_name M2_SEQ_step2_fetch_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [5/5] M2 joint multitask on BabyAI-Pickup-v0 + MiniGrid-Fetch-8x8-N3-v0
echo ============================================================
%PYTHON_BIN% .\train_sb3_gc_ppo_multitask.py --env_ids BabyAI-Pickup-v0,MiniGrid-Fetch-8x8-N3-v0 --module_name M2_Pickup_joint --parent_goal near_obj --child_goal has_obj --goals has_obj --option_horizon 256 --total_timesteps 220000 --n_envs 8 --seed 0 --no_strict_parent_fail_fast --task_sampling uniform --run_name M2_joint_nff_s0
if errorlevel 1 goto :fail

echo.
echo All M2 runs completed successfully.
echo.
echo Experts:
echo   runs_gcppo\M2_E_babyaipickup_nff_s0\
echo   runs_gcppo\M2_E_fetch_nff_s0\
echo.
echo Sequential shared:
echo   runs_gcppo\M2_SEQ_step1_babyaipickup_nff_s0\
echo   runs_gcppo\M2_SEQ_step2_fetch_nff_s0\
echo.
echo Joint shared:
echo   runs_gcppo\M2_joint_nff_s0\
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1