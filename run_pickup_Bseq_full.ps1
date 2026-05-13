$ErrorActionPreference = "Continue"
$env:PYTHONWARNINGS = "ignore::UserWarning:pygame.pkgdata"

cd D:\Project\HRC_granularity

# ============================================================
# G2 Pickup Skill Family: B Sequential Aggregation
# Task order:
#   MiniGrid-Fetch-8x8-N3-v0
#       -> BabyAI-PickupLoc-v0
#       -> BabyAI-Pickup-v0
#
# Skills:
#   approach_target
#   pickup_target
# ============================================================

New-Item -ItemType Directory -Force -Path runs_pickup_bseq | Out-Null
New-Item -ItemType Directory -Force -Path logs_pickup_bseq | Out-Null
New-Item -ItemType Directory -Force -Path results_pickup_bseq | Out-Null

# Environment check
python -c "import gymnasium as gym; import minigrid; required=['MiniGrid-Fetch-8x8-N3-v0','BabyAI-PickupLoc-v0','BabyAI-Pickup-v0']; missing=[e for e in required if e not in gym.envs.registry.keys()]; print('Missing:', missing); assert not missing"

function Run-Train {
    param (
        [string]$RunName,
        [string]$EnvId,
        [string]$Template,
        [string]$ParentGoal,
        [int]$TotalTimesteps,
        [int]$OptionHorizon,
        [string]$LoadModel
    )

    $RunDir = Join-Path "runs_pickup_bseq" $RunName
    $ExpectedModel = Join-Path $RunDir ("{0}_final.zip" -f $EnvId)

    if (Test-Path $ExpectedModel) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "SKIP existing: $RunName"
        Write-Host "Found: $ExpectedModel"
        Write-Host "============================================================"
        return
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running:  $RunName"
    Write-Host "Env:      $EnvId"
    Write-Host "Template: $Template"
    Write-Host "Steps:    $TotalTimesteps"
    if ($LoadModel -ne "") {
        Write-Host "Load:     $LoadModel"
    }
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
        "--logdir", "runs_pickup_bseq",
        "--run_name", $RunName
    )

    if ($ParentGoal -ne "") {
        $args += @("--parent_goal", $ParentGoal)
    }

    if ($LoadModel -ne "") {
        $args += @("--load_model", $LoadModel)
    }

    python @args 2>&1 | Tee-Object -FilePath "logs_pickup_bseq\$RunName.log"
}

# ============================================================
# A. approach_target: Fetch -> PickupLoc -> BabyAI-Pickup
# ============================================================

Run-Train `
    -RunName "PICK_Bseq_01_fetch_approach_target_s0" `
    -EnvId "MiniGrid-Fetch-8x8-N3-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 300000 `
    -OptionHorizon 256 `
    -LoadModel ""

Run-Train `
    -RunName "PICK_Bseq_02_fetch_pickuploc_approach_target_s0" `
    -EnvId "BabyAI-PickupLoc-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 400000 `
    -OptionHorizon 256 `
    -LoadModel "runs_pickup_bseq\PICK_Bseq_01_fetch_approach_target_s0\MiniGrid-Fetch-8x8-N3-v0_final.zip"

Run-Train `
    -RunName "PICK_Bseq_03_fetch_pickuploc_babyai_approach_target_s0" `
    -EnvId "BabyAI-Pickup-v0" `
    -Template "approach_target" `
    -ParentGoal "" `
    -TotalTimesteps 500000 `
    -OptionHorizon 256 `
    -LoadModel "runs_pickup_bseq\PICK_Bseq_02_fetch_pickuploc_approach_target_s0\BabyAI-PickupLoc-v0_final.zip"

# ============================================================
# B. pickup_target: Fetch -> PickupLoc -> BabyAI-Pickup
# ============================================================

Run-Train `
    -RunName "PICK_Bseq_01_fetch_pickup_target_s0" `
    -EnvId "MiniGrid-Fetch-8x8-N3-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128 `
    -LoadModel ""

Run-Train `
    -RunName "PICK_Bseq_02_fetch_pickuploc_pickup_target_s0" `
    -EnvId "BabyAI-PickupLoc-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128 `
    -LoadModel "runs_pickup_bseq\PICK_Bseq_01_fetch_pickup_target_s0\MiniGrid-Fetch-8x8-N3-v0_final.zip"

Run-Train `
    -RunName "PICK_Bseq_03_fetch_pickuploc_babyai_pickup_target_s0" `
    -EnvId "BabyAI-Pickup-v0" `
    -Template "pickup_target" `
    -ParentGoal "near_target" `
    -TotalTimesteps 300000 `
    -OptionHorizon 128 `
    -LoadModel "runs_pickup_bseq\PICK_Bseq_02_fetch_pickuploc_pickup_target_s0\BabyAI-PickupLoc-v0_final.zip"

Write-Host ""
Write-Host "============================================================"
Write-Host "Finished G2 pickup B sequential aggregation:"
Write-Host "  Fetch -> PickupLoc -> BabyAI-Pickup"
Write-Host ""
Write-Host "Final approach_target model:"
Write-Host "  runs_pickup_bseq\PICK_Bseq_03_fetch_pickuploc_babyai_approach_target_s0\BabyAI-Pickup-v0_final.zip"
Write-Host ""
Write-Host "Final pickup_target model:"
Write-Host "  runs_pickup_bseq\PICK_Bseq_03_fetch_pickuploc_babyai_pickup_target_s0\BabyAI-Pickup-v0_final.zip"
Write-Host "============================================================"