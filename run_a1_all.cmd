@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/6] A1 expert on MiniGrid-DoorKey-6x6-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-DoorKey-6x6-v0 --goals has_key --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A1_E_doorkey_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/6] A1 expert on MiniGrid-UnlockPickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-UnlockPickup-v0 --goals has_key --option_horizon 256 --total_timesteps 120000 --seed 0 --run_name A1_E_unlockpickup_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/6] A1 expert on BabyAI-UnlockLocal-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-UnlockLocal-v0 --goals has_key --option_horizon 256 --total_timesteps 120000 --seed 0 --run_name A1_E_unlocklocal_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/6] A1 sequential step1 on MiniGrid-DoorKey-6x6-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-DoorKey-6x6-v0 --goals has_key --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A1_SEQ_step1_doorkey_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [5/6] A1 sequential step2 on MiniGrid-UnlockPickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-UnlockPickup-v0 --goals has_key --option_horizon 256 --total_timesteps 80000 --seed 0 --load_model runs_gcppo\A1_SEQ_step1_doorkey_s0\MiniGrid-DoorKey-6x6-v0_final.zip --run_name A1_SEQ_step2_unlockpickup_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [6/6] A1 sequential step3 on BabyAI-UnlockLocal-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-UnlockLocal-v0 --goals has_key --option_horizon 256 --total_timesteps 80000 --seed 0 --load_model runs_gcppo\A1_SEQ_step2_unlockpickup_s0\MiniGrid-UnlockPickup-v0_final.zip --run_name A1_SEQ_step3_unlocklocal_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [JOINT] A1 joint multitask
echo ============================================================
%PYTHON_BIN% .\train_sb3_gc_ppo_multitask.py --env_ids MiniGrid-DoorKey-6x6-v0,MiniGrid-UnlockPickup-v0,BabyAI-UnlockLocal-v0 --module_name A1_AcquireKey_joint --goals has_key --option_horizon 256 --total_timesteps 220000 --n_envs 8 --seed 0 --task_sampling uniform --run_name A1_joint_s0
if errorlevel 1 goto :fail

echo.
echo All A1 runs completed successfully.
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1