# Untrack .jpg / .jpeg from git index (files stay on disk).
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\untrack_jpg_from_git.ps1

$ErrorActionPreference = "Stop"

if (-not $PSScriptRoot) {
    Write-Error "PSScriptRoot missing; run with -File."
    exit 1
}
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

function Find-GitExe {
    foreach ($name in @('git.exe', 'git')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -and (Test-Path -LiteralPath $cmd.Source)) {
            return $cmd.Source
        }
    }
    $whereExe = Join-Path $env:SystemRoot 'System32\where.exe'
    if (Test-Path -LiteralPath $whereExe) {
        try {
            $fromWhere = & $whereExe git 2>$null | Where-Object { $_ -and ($_ -match '\.exe$') } | Select-Object -First 1
            if ($fromWhere -and (Test-Path -LiteralPath $fromWhere.Trim())) {
                return $fromWhere.Trim()
            }
        } catch {}
    }
    foreach ($p in @(
        "${env:ProgramFiles}\Git\cmd\git.exe",
        "${env:ProgramFiles}\Git\bin\git.exe",
        "${env:ProgramFiles(x86)}\Git\cmd\git.exe",
        "${env:ProgramFiles(x86)}\Git\bin\git.exe",
        "${env:LocalAppData}\Programs\Git\cmd\git.exe",
        "${env:LocalAppData}\Programs\Git\bin\git.exe",
        "${env:USERPROFILE}\scoop\apps\git\current\bin\git.exe",
        "${env:USERPROFILE}\scoop\shims\git.exe",
        "${env:ProgramFiles}\Git\mingw64\bin\git.exe",
        "${env:ProgramFiles(x86)}\Git\mingw64\bin\git.exe"
    )) {
        if ($p -and (Test-Path -LiteralPath $p)) { return $p }
    }
    return $null
}

$gitExe = Find-GitExe
if (-not $gitExe) {
    Write-Error @"
git.exe not found.

Install Git for Windows (https://git-scm.com/download/win) and reopen the terminal,
or use "Git Bash" / GitHub Desktop from the Start menu and run git commands there.

If Git is already installed, add its bin folder to PATH (often:
  C:\Program Files\Git\cmd
)
"@
    exit 1
}

$allFiles = & $gitExe -C $repoRoot ls-files 2>$null
if ($null -eq $allFiles) {
    $allFiles = @()
} elseif ($allFiles -is [string]) {
    $allFiles = @($allFiles)
}

$jpgFiles = @(
    $allFiles | Where-Object -FilterScript {
        $line = $_
        $line -match '\.(jpe?g)$'
    }
)

if ($jpgFiles.Count -eq 0) {
    Write-Host "No tracked .jpg / .jpeg files."
    exit 0
}

Write-Host "Removing $($jpgFiles.Count) path(s) from index (keeping files on disk):"
foreach ($line in $jpgFiles) {
    Write-Host "  $line"
}

foreach ($rel in $jpgFiles) {
    & $gitExe -C $repoRoot rm --cached -- $rel
}

$dq = [char]34
$commitMsg = "chore: stop tracking jpg/jpeg images"
Write-Host ""
Write-Host "Done. Next:"
Write-Host "  git status"
Write-Host ("  git commit -m " + $dq + $commitMsg + $dq)
Write-Host "  git push"
