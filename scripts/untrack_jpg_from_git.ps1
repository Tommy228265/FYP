# 从 Git 索引中移除所有已跟踪的 .jpg / .jpeg（文件仍保留在磁盘上）。
# 用法：在资源管理器中右键「使用 PowerShell 运行」，或在仓库根目录执行：
#   powershell -ExecutionPolicy Bypass -File scripts\untrack_jpg_from_git.ps1
# 然后执行：
#   git commit -m "chore: stop tracking jpg/jpeg images"
#   git push

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

function Find-GitExe {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "${env:ProgramFiles}\Git\cmd\git.exe",
        "${env:ProgramFiles(x86)}\Git\cmd\git.exe",
        "${env:LocalAppData}\Programs\Git\cmd\git.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$git = Find-GitExe
if (-not $git) {
    Write-Error "未找到 git.exe，请先安装 Git for Windows 并加入 PATH。"
    exit 1
}

$tracked = @(& $git ls-files | Where-Object { $_ -match '\.(jpe?g)$' })
if ($tracked.Count -eq 0) {
    Write-Host "当前没有已被 Git 跟踪的 .jpg / .jpeg 文件。"
    exit 0
}

Write-Host "将从 Git 索引移除 $($tracked.Count) 个文件（不删除本地文件）："
$tracked | ForEach-Object { Write-Host "  $_" }

foreach ($f in $tracked) {
    & $git rm --cached -- $f
}

Write-Host ""
Write-Host "完成。请执行："
Write-Host "  git status"
Write-Host "  git commit -m \"chore: stop tracking jpg/jpeg images\""
Write-Host "  git push"
