# Keeps run 3 alive: every minute, if the run is not running and not finished, start it again (it resumes).
# Does not restart after a stage hit its step cap (outputs\run3\FAILED exists): that needs a decision.
# Also logs when no new training data has appeared for 25 minutes while the process is alive. Log: outputs\run3_watchdog.log
$root = Split-Path $PSScriptRoot -Parent
$log = "$root\outputs\run3_watchdog.log"
function Note($m) { Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }
Note "watchdog started"
$stalled = $false
$told = $false
while ($true) {
    if (Test-Path "$root\outputs\run3\COMPLETE") { Note "run complete, watchdog exiting"; break }
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run3_train\.py' }
    if (-not $running) {
        if (Test-Path "$root\outputs\run3\FAILED") {
            if (-not $told) { Note "STOPPED: a stage hit its step cap (outputs\run3\FAILED); not restarting, needs a decision"; $told = $true }
            Start-Sleep 300
            continue
        }
        Note "RESTART: run3_train.py was not running"
        & powershell -File "$root\scripts\start_run3.ps1" | Out-Null
        Start-Sleep 120
        continue
    }
    $told = $false
    $newest = Get-ChildItem "$root\outputs\run3\*\train_eps\*.npz" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest -and ((Get-Date) - $newest.LastWriteTime).TotalMinutes -gt 25) {
        $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run3_gate_eval' }
        if (-not $busy -and -not $stalled) { Note "STALL: no new training data for 25+ minutes"; $stalled = $true }
    } else { $stalled = $false }
    Start-Sleep 60
}
