@echo off
setlocal

set PYTHON_BIN=python

echo.
echo ============================================================
echo [1/4] A0 OpenBox on BabyAI-KeyInBox-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-KeyInBox-v0 --goals open_box --option_horizon 256 --total_timesteps 100000 --seed 0 --run_name A0_OpenBox_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [2/4] M3 OpenBoxKey on BabyAI-KeyInBox-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id BabyAI-KeyInBox-v0 --module_name M3_OpenBoxKey --parent_goal open_box --child_goal has_key --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 160000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --run_name M3_OpenBoxKey_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [3/4] M4 DoorToBox on MiniGrid-UnlockPickup-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-UnlockPickup-v0 --module_name M4_DoorToBox --parent_goal door_open --child_goal has_box --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 160000 --seed 0 --strict_reset_max_tries 128 --no_strict_parent_fail_fast --run_name M4_DoorToBox_s0
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo [4/4] M5 DoorToGoal on MiniGrid-DoorKey-6x6-v0
echo ============================================================
%PYTHON_BIN% -m rl.train_sb3_gc_ppo --env_id MiniGrid-DoorKey-6x6-v0 --module_name M5_DoorToGoal --parent_goal door_open --child_goal at_goal --edge_goal_pool_mode child_only --option_horizon 256 --total_timesteps 140000 --seed 0 --no_strict_parent_fail_fast --run_name M5_DoorToGoal_s0
if errorlevel 1 goto :fail

echo.
echo All single-task modules completed successfully.
goto :eof

:fail
echo.
echo ERROR: A command failed. Stop here and check the output above.
exit /b 1