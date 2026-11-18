<#
Starts (or resumes) run 3 as a hidden background process that is independent of this terminal and of any chat session.
Safe to run again at any time: it only starts the run if it is not already running, and a restarted run resumes from the
last checkpoint of the stage it was in (finished stages are skipped).

Progress:  .venv\Scripts\python.exe scripts\run3_status.py
Log:       outputs\run3_console.log

Usage (from the project root):   powershell -File scripts\start_run3.ps1 [-Base outputs/run3]
#>
param([string]$Base = 'outputs/run3')

$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\outputs" | Out-Null

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run3_train\.py' }
if ($running) {
    Write-Host "Run 3 is already running (process $($running[0].ProcessId))."
} else {
    $cmd = ".venv\Scripts\python.exe -u scripts\run3_train.py --base $Base >> outputs\run3_console.log 2>&1"
    Start-Process -FilePath cmd.exe -WindowStyle Hidden -WorkingDirectory $root -ArgumentList '/c', $cmd
    Write-Host "Run 3 started in the background. Log: outputs\run3_console.log"
}
Write-Host "Check on it with:  .venv\Scripts\python.exe scripts\run3_status.py"
