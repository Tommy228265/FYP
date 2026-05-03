# Train PhysFormer on UBFC-rPPG. Example:
#   $env:UBFC_ROOT = "D:\datasets\UBFC-rPPG"
#   .\run_train_ubfc.ps1

param(
    [string]$UbfcRoot = "",
    [int]$Gpu = 0,
    [int]$Epochs = 25,
    [string]$LogName = "Physformer_UBFC_run1",
    [string]$Pretrained = ""
)

$root = $UbfcRoot
if (-not $root) { $root = $env:UBFC_ROOT }
if (-not $root) {
    Write-Host "Set UBFC_ROOT to the folder containing subject*/vid.avi and ground_truth.txt, e.g.:" -ForegroundColor Yellow
    Write-Host '  $env:UBFC_ROOT = "D:\datasets\UBFC-rPPG"' -ForegroundColor Cyan
    exit 1
}

$repo = Join-Path $PSScriptRoot "PhysFormer"
Set-Location $repo

$preArg = @()
if ($Pretrained) {
    $preArg = @("--pretrained", $Pretrained)
}

python train_Physformer_160_UBFC.py --ubfc_root $root --gpu $Gpu --epochs $Epochs --log $LogName @preArg
