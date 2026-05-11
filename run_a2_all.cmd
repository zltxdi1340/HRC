@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/6] A2 expert on MiniGrid-GoToObject-8x8-N2-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-GoToObject-8x8-N2-v0 --goals near_obj --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A2_E_goto_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/6] A2 expert on BabyAI-Pickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-Pickup-v0 --goals near_obj --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A2_E_babyaipickup_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/6] A2 expert on MiniGrid-Fetch-8x8-N3-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Fetch-8x8-N3-v0 --goals near_obj --option_horizon 256 --total_timesteps 120000 --seed 0 --run_name A2_E_fetch_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/6] A2 sequential step1 on MiniGrid-GoToObject-8x8-N2-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-GoToObject-8x8-N2-v0 --goals near_obj --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A2_SEQ_step1_goto_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [5/6] A2 sequential step2 on BabyAI-Pickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-Pickup-v0 --goals near_obj --option_horizon 256 --total_timesteps 80000 --seed 0 --load_model runs_gcppo\A2_SEQ_step1_goto_s0\MiniGrid-GoToObject-8x8-N2-v0_final.zip --run_name A2_SEQ_step2_babyaipickup_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [6/6] A2 sequential step3 on MiniGrid-Fetch-8x8-N3-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Fetch-8x8-N3-v0 --goals near_obj --option_horizon 256 --total_timesteps 80000 --seed 0 --load_model runs_gcppo\A2_SEQ_step2_babyaipickup_s0\BabyAI-Pickup-v0_final.zip --run_name A2_SEQ_step3_fetch_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [JOINT] A2 joint multitask
echo ============================================================
%PYTHON_BIN% .\train_sb3_gc_ppo_multitask.py --env_ids MiniGrid-GoToObject-8x8-N2-v0,BabyAI-Pickup-v0,MiniGrid-Fetch-8x8-N3-v0 --module_name A2_ApproachObj_joint --goals near_obj --option_horizon 256 --total_timesteps 240000 --n_envs 8 --seed 0 --task_sampling uniform --run_name A2_joint_s0
if errorlevel 1 goto :fail

echo.
echo All A2 runs completed successfully.
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1