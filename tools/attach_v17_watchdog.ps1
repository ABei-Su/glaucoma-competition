$ErrorActionPreference='Stop'
$root='E:/bisai/glaucoma_project'
trap { $_ | Out-String | Add-Content -LiteralPath "$root/reports/v17_attached_watcher.stderr"; exit 1 }
Set-Location $root
$review=Get-Content reports/v17_watchdog_review.json -Raw | ConvertFrom-Json
if(-not $review.approved) {throw 'Watch attachment requires review'}
foreach($f in $review.files) {if((Get-FileHash -LiteralPath $f.path -Algorithm SHA256).Hash.ToLower() -ne $f.sha256) {throw 'Changed repair input'}}
$launch=Get-Content reports/v17_launch.json -Raw | ConvertFrom-Json
if(Get-CimInstance Win32_Process -Filter "ProcessId = $($launch.watcher_pid)") {throw 'Original watcher still active'}
if(Test-Path reports/v17_exit.json) {throw 'Training exit already recorded'}
if(Test-Path output/v17/finalization.reservation) {throw 'Finalization already reserved'}
$proc=Get-CimInstance Win32_Process -Filter "ProcessId = $($launch.pid)"
if(-not $proc -or $proc.Name -ne 'python.exe' -or $proc.CommandLine -notmatch 'train.py.*--run runs/v17_r50_pos2_8.*--pos-weight 2') {throw 'Trainer identity mismatch'}
$p=[Diagnostics.Process]::GetProcessById($launch.pid)
$null=$p.Handle
if([Math]::Abs(($p.StartTime.ToUniversalTime()-[DateTimeOffset]::Parse($launch.started_at).UtcDateTime).TotalSeconds) -gt 30) {throw 'Trainer PID start-time mismatch'}
$freeze=Get-Content output/v17/frozen_inputs.json -Raw | ConvertFrom-Json
foreach($f in $freeze.files) {if((Get-FileHash -LiteralPath $f.path -Algorithm SHA256).Hash.ToLower() -ne $f.sha256) {throw 'Frozen input changed'}}
$reservation=[IO.File]::Open("$root/output/v17/watchdog_repair_001/attachment.reservation",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try {$b=[Text.Encoding]::UTF8.GetBytes([string]$PID);$reservation.Write($b,0,$b.Length)} finally {$reservation.Dispose()}
[pscustomobject]@{pid=$p.Id;watcher_pid=$PID;attached_at=(Get-Date).ToString('o');deadline=$launch.deadline;training_restarted=$false} | ConvertTo-Json | Set-Content reports/v17_attached_watcher.json -Encoding utf8
$deadline=[DateTimeOffset]::Parse($launch.deadline)
$remaining=[Math]::Max(0,[int]($deadline-[DateTimeOffset]::Now).TotalMilliseconds)
if(-not $p.WaitForExit($remaining)) {
 & taskkill /PID $p.Id /T /F
 [pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='budget_exceeded';pid=$p.Id;kill_exit=$LASTEXITCODE;watcher='attached'} | ConvertTo-Json | Set-Content reports/v17_exit.json -Encoding utf8
 exit 1
}
$p.Refresh()
$code=$p.ExitCode
if($null -eq $code) {throw 'Trainer exit code unavailable; do not fabricate success'}
[pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='exited';pid=$p.Id;exit_code=$code;watcher='attached'} | ConvertTo-Json | Set-Content reports/v17_exit.json -Encoding utf8
if($code -ne 0) {exit $code}
& 'C:/Users/SuYanPei/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe' -NoProfile -File "$root/tools/launch_v17_finalizer.ps1"
exit $LASTEXITCODE
