$ErrorActionPreference = "Stop"

# =========================
# Directories
# =========================

New-Item -ItemType Directory -Force -Path runs_pickup_chain | Out-Null
New-Item -ItemType Directory -Force -Path results_pickup_chain | Out-Null
New-Item -ItemType Directory -Force -Path results_pickup_chain\source | Out-Null
New-Item -ItemType Directory -Force -Path results_pickup_chain\transfer | Out-Null
New-Item -ItemType Directory -Force -Path logs_pickup_chain | Out-Null

# =========================
# Environment check
# =========================

python -c "import gymnasium as gym; import minigrid; required=['BabyAI-PickupLoc-v0','BabyAI-Pickup-v0','MiniGrid-Fetch-8x8-N3-v0']; missing=[e for e in required if e not in gym.envs.registry.keys()]; print('Missing:', missing); assert not missing"

# =========================
# Helper
# =========================

function Run-Train {
    param (
        [string]$RunName,
        [string]$EnvId,
        [string]$Template,
        [string]$ParentGoal,
        [int]$TotalTimesteps,
        [int]$OptionHorizon
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running: $RunName"
    Write-Host "Env:     $EnvId"
    Write-Host "Template:$Template"
    Write-Host "Steps:   $TotalTimesteps"
    Write-Host "============================================================"
    Write-Host ""

    $args = @(
        "-m", "rl.train_sb3_g2_ppo",
        "--env_id", $EnvId,
        "--template", $Template,
        "--total_timesteps", "$TotalTimesteps",
        "--n_envs", "8",
        "--seed", "0",
        "--option_horizon", "$OptionHorizon",
        "--logdir", "runs_pickup_chain",
        "--run_name", $RunName
    )

    if ($ParentGoal -ne "") {
        $args += @("--parent_goal", $ParentGoal)
    }

    python -W ignore @args 2>&1 | Tee-Object -FilePath "logs_pickup_chain\$RunName.log"
}

# =========================
# 1. Fetch source task
# =========================

Run-Train `
    -RunName "PICK_fetch8x8n3_approach_target_s0" `
    -EnvId "MiniGrid-Fetch-8x8-N3-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 300000 `
    -OptionHorizon 256

Run-Train `
    -RunName "PICK_fetch8x8n3_pickup_target_s0" `
    -EnvId "MiniGrid-Fetch-8x8-N3-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128

# =========================
# 2. BabyAI-PickupLoc source task
# =========================

Run-Train `
    -RunName "PICK_pickuploc_approach_target_s0" `
    -EnvId "BabyAI-PickupLoc-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 400000 `
    -OptionHorizon 256

Run-Train `
    -RunName "PICK_pickuploc_pickup_target_s0" `
    -EnvId "BabyAI-PickupLoc-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128

# =========================
# 3. BabyAI-Pickup target task direct training
#    This is not the transfer result.
#    It is used as direct target-task reference / upper bound.
# =========================

Run-Train `
    -RunName "PICK_babyaipickup_approach_target_s0" `
    -EnvId "BabyAI-Pickup-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 500000 `
    -OptionHorizon 256

Run-Train `
    -RunName "PICK_babyaipickup_pickup_target_s0" `
    -EnvId "BabyAI-Pickup-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128

Write-Host ""
Write-Host "============================================================"
Write-Host "All pickup-chain trainings finished."
Write-Host "Models saved under: D:\Project\HRC_granularity\runs_pickup_chain"
Write-Host "Logs saved under:   D:\Project\HRC_granularity\logs_pickup_chain"
Write-Host "============================================================"