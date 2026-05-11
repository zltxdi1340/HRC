@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/8] M1 expert on MiniGrid-DoorKey-6x6-v0 (no fail-fast)
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-DoorKey-6x6-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 120000 --seed 0 --no_strict_parent_fail_fast --run_name M1_E_doorkey_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/8] M1 expert on MiniGrid-UnlockPickup-v0 (no fail-fast)
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-UnlockPickup-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 140000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --run_name M1_E_unlockpickup_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/8] M1 expert on BabyAI-UnlockLocal-v0 (no fail-fast)
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-UnlockLocal-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 140000 --seed 0 --no_strict_parent_fail_fast --run_name M1_E_unlocklocal_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/8] M1 expert on BabyAI-KeyInBox-v0 (no fail-fast)
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-KeyInBox-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 180000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --run_name M1_E_keyinbox_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [5/8] Shared M1 step1 on MiniGrid-DoorKey-6x6-v0 (base)
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-DoorKey-6x6-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 120000 --seed 0 --no_strict_parent_fail_fast --run_name M1_SEQ_step1_doorkey_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [6/8] Shared M1 step2 on MiniGrid-UnlockPickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-UnlockPickup-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 80000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --load_model runs_gcppo\M1_SEQ_step1_doorkey_nff_s0\MiniGrid-DoorKey-6x6-v0_final.zip --run_name M1_SEQ_step2_unlockpickup_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [7/8] Shared M1 step3 on BabyAI-UnlockLocal-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-UnlockLocal-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 80000 --seed 0 --no_strict_parent_fail_fast --load_model runs_gcppo\M1_SEQ_step2_unlockpickup_nff_s0\MiniGrid-UnlockPickup-v0_final.zip --run_name M1_SEQ_step3_unlocklocal_nff_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [8/8] Shared M1 step4 on BabyAI-KeyInBox-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-KeyInBox-v0 --module_name M1_UnlockDoor --parent_goal has_key --child_goal door_open --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 100000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --load_model runs_gcppo\M1_SEQ_step3_unlocklocal_nff_s0\BabyAI-UnlockLocal-v0_final.zip --run_name M1_SEQ_step4_keyinbox_nff_s0
if errorlevel 1 goto :fail

echo.
echo All M1 no-fail-fast runs completed successfully.
echo.
echo Experts:
echo   runs_gcppo\M1_E_doorkey_nff_s0\
echo   runs_gcppo\M1_E_unlockpickup_nff_s0\
echo   runs_gcppo\M1_E_unlocklocal_nff_s0\
echo   runs_gcppo\M1_E_keyinbox_nff_s0\
echo.
echo Shared sequential:
echo   runs_gcppo\M1_SEQ_step1_doorkey_nff_s0\
echo   runs_gcppo\M1_SEQ_step2_unlockpickup_nff_s0\
echo   runs_gcppo\M1_SEQ_step3_unlocklocal_nff_s0\
echo   runs_gcppo\M1_SEQ_step4_keyinbox_nff_s0\
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1