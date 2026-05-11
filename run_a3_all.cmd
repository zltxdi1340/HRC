@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/5] A3 expert on MiniGrid-Empty-8x8-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Empty-8x8-v0 --goals at_goal --option_horizon 256 --total_timesteps 80000 --seed 0 --run_name A3_E_empty_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/5] A3 expert on MiniGrid-FourRooms-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-FourRooms-v0 --goals at_goal --option_horizon 512 --total_timesteps 120000 --seed 0 --run_name A3_E_fourrooms_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/5] A3 sequential step1 on MiniGrid-Empty-8x8-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-Empty-8x8-v0 --goals at_goal --option_horizon 256 --total_timesteps 80000 --seed 0 --run_name A3_SEQ_step1_empty_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/5] A3 sequential step2 on MiniGrid-FourRooms-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-FourRooms-v0 --goals at_goal --option_horizon 512 --total_timesteps 80000 --seed 0 --load_model runs_gcppo\A3_SEQ_step1_empty_s0\MiniGrid-Empty-8x8-v0_final.zip --run_name A3_SEQ_step2_fourrooms_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [5/5] A3 joint multitask
echo ============================================================
%PYTHON_BIN% .\train_sb3_gc_ppo_multitask.py --env_ids MiniGrid-Empty-8x8-v0,MiniGrid-FourRooms-v0 --module_name A3_ReachGoal_joint --goals at_goal --option_horizon 256 --total_timesteps 180000 --n_envs 8 --seed 0 --task_sampling uniform --run_name A3_joint_s0
if errorlevel 1 goto :fail

echo.
echo All A3 runs completed successfully.
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1
