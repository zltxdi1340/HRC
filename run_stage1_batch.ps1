$PYTHON_BIN = "python"
$OUT_DIR = "outputs/stage1"
$SEED = 0

New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

function Run-One {
    param(
        [string]$env_id,
        [string]$name,
        [int]$T,
        [int]$H,
        [int]$delta,
        [double]$tau,
        [int]$min_support,
        [switch]$recheck_roots
    )

    $summary_path = "$OUT_DIR/${name}_summary.json"

    Write-Host "============================================================"
    Write-Host "[RUN] env=$env_id"
    Write-Host "      summary=$summary_path"
    Write-Host "      T=$T H=$H delta=$delta tau=$tau min_support=$min_support seed=$SEED"
    Write-Host "============================================================"

    $args = @(
        "hrc_stage1.py",
        "--env_id", $env_id,
        "--T", "$T",
        "--H", "$H",
        "--delta", "$delta",
        "--tau", "$tau",
        "--min_support", "$min_support",
        "--seed", "$SEED"
    )

    if ($recheck_roots) {
        $args += "--recheck_roots"
    }

    $args += @("--out_summary", $summary_path)

    & $PYTHON_BIN @args
    if ($LASTEXITCODE -ne 0) {
        throw "hrc_stage1.py 运行失败: $env_id"
    }

    # Write-Host "[VIZ] $summary_path"
    # & $PYTHON_BIN "viz/viz_graph_slices.py" --summary $summary_path
    # if ($LASTEXITCODE -ne 0) {
    #     throw "viz_graph_slices.py 运行失败: $summary_path"
    # }

    Write-Host "[DONE] $env_id"
    Write-Host ""
}

Run-One "MiniGrid-Empty-8x8-v0"         "empty"         40  80  20 0.05 40
Run-One "MiniGrid-FourRooms-v0"         "fourrooms"     40  80  20 0.05 40
Run-One "MiniGrid-GoToObject-8x8-N2-v0" "gotoobject"    40  80  20 0.05 40
Run-One "BabyAI-UnlockLocal-v0"         "unlocklocal"  100 150  40 0.05 40 -recheck_roots
Run-One "MiniGrid-DoorKey-6x6-v0"       "doorkey"      100 150  40 0.05 40 -recheck_roots
Run-One "MiniGrid-UnlockPickup-v0"      "unlockpickup" 100 150  40 0.05 40 -recheck_roots
Run-One "MiniGrid-Fetch-8x8-N3-v0"      "fetch"        100 150  40 0.05 40 -recheck_roots
Run-One "BabyAI-Pickup-v0"              "babyaipickup" 100 150  40 0.05 40 -recheck_roots
Run-One "BabyAI-KeyInBox-v0"            "keyinbox"     120 180  50 0.05 40 -recheck_roots

Write-Host "全部完成。结果在: $OUT_DIR"