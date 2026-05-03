# Train PhysFormer on VIPL-HR (after you have prepared VIPL_frames under $env:VIPL_PARENT).
# Usage:
#   $env:VIPL_PARENT = "D:\datasets\PhysFormer_VIPL"
#   .\run_train_vipl.ps1
# VIPL_PARENT must contain a subdirectory named exactly: VIPL_frames

param(
    [int]$Gpu = 0,
    [int]$Epochs = 25,
    [string]$LogName = "Physformer_VIPL_fold1_run1"
)

$parent = $env:VIPL_PARENT
if (-not $parent) {
    Write-Host "Set VIPL_PARENT to the folder that CONTAINS VIPL_frames, e.g.:" -ForegroundColor Yellow
    Write-Host '  $env:VIPL_PARENT = "D:\datasets\PhysFormer_VIPL"' -ForegroundColor Cyan
    exit 1
}

$frames = Join-Path $parent "VIPL_frames"
if (-not (Test-Path $frames)) {
    Write-Host "Missing folder: $frames" -ForegroundColor Red
    Write-Host "VIPL_PARENT should be the parent directory of VIPL_frames." -ForegroundColor Yellow
    exit 1
}

$repo = Join-Path $PSScriptRoot "PhysFormer"
Set-Location $repo

python train_Physformer_160_VIPL.py --input_data $parent --gpu $Gpu --epochs $Epochs --log $LogName
