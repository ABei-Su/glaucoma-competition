$ErrorActionPreference='Stop'
$root='E:/bisai/glaucoma_project'
trap { $_ | Out-String | Add-Content -LiteralPath "$root/reports/v15_recovery_watcher.stderr"; exit 1 }
Set-Location $root
if(Test-Path runs/v15_full_r50_8) {throw 'Preserve existing run'}
if(Test-Path reports/v15_recovery_launch.json) {throw 'Duplicate recovery forbidden'}
$original=Get-Content reports/v15_launch.json -Raw | ConvertFrom-Json
$deadline=[DateTimeOffset]::Parse($original.deadline)
$remaining=[int]($deadline-[DateTimeOffset]::Now).TotalMilliseconds
if($remaining -le 0) {throw 'Original deadline expired'}
if(Get-CimInstance Win32_Process | Where-Object {$_.ProcessId -in @($original.pid,$original.watcher_pid)}) {throw 'Original processes still active'}
$recoveryReview=Get-Content reports/v15_recovery_review.json -Raw | ConvertFrom-Json
if(-not $recoveryReview.approved) {throw 'Recovery review required'}
foreach($f in $recoveryReview.files) {if((Get-FileHash -LiteralPath $f.path -Algorithm SHA256).Hash.ToLower() -ne $f.sha256) {throw 'Changed recovery source'}}
$review=Get-Content reports/v15_preflight_review.json -Raw | ConvertFrom-Json
if(-not $review.approved) {throw 'Independent review required'}
$freeze=Get-Content output/v15/frozen_inputs.json -Raw | ConvertFrom-Json
foreach($entry in $freeze.files) {if((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) {throw "Frozen input changed: $($entry.path)"}}
# Atomic reservation persists as audit evidence; no automatic cleanup/retry.
$reservation=[IO.File]::Open("$root/output/v15/recovery_001/recovery.launch.reservation",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try { $bytes=[Text.Encoding]::UTF8.GetBytes([string]$PID); $reservation.Write($bytes,0,$bytes.Length) } finally { $reservation.Dispose() }
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP="$root/tmp"
$env:TMP=$env:TEMP
$env:TORCH_HOME="$root/pretrain_model"
$env:HF_HOME="$root/tmp/hf"
$env:XDG_CACHE_HOME="$root/tmp/cache"
$env:CUDA_VISIBLE_DEVICES='0'
$python='C:/Users/SuYanPei/AppData/Local/Programs/Python/Python39/python.exe'
$remaining=[int]($deadline-[DateTimeOffset]::Now).TotalMilliseconds
if($remaining -le 0) {throw 'Original deadline expired before launch'}
$p=Start-Process -FilePath $python -ArgumentList @('-u','train.py','--architecture','resnet50','--cache','data/cache384','--run','runs/v15_full_r50_8','--epochs','8','--checkpoint-epoch','8','--size','384','--batch','16','--workers','4','--seed','20261032','--lr','0.0001','--pos-weight','1','--augmentation','mild','--validation-probability-precision','fp32') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root/reports/v15_train_recovery001.log" -RedirectStandardError "$root/reports/v15_train_recovery001.stderr" -PassThru
[pscustomobject]@{pid=$p.Id;watcher_pid=$PID;started_at=(Get-Date).ToString('o');deadline=$original.deadline;plan='docs/science-superpowers/plans/2026-10-30-final-sprint.md'} | ConvertTo-Json | Set-Content reports/v15_recovery_launch.json -Encoding UTF8
$remaining=[Math]::Max(0,[int]($deadline-[DateTimeOffset]::Now).TotalMilliseconds)
$finished=$p.WaitForExit($remaining)
if(-not $finished) {
 & taskkill /PID $p.Id /T /F
 [pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='budget_exceeded';pid=$p.Id;kill_exit=$LASTEXITCODE} | ConvertTo-Json | Set-Content reports/v15_exit.json -Encoding UTF8
 exit 1
}
$p.Refresh()
[pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='exited';pid=$p.Id;exit_code=$p.ExitCode} | ConvertTo-Json | Set-Content reports/v15_exit.json -Encoding UTF8
if($p.ExitCode -ne 0) {exit $p.ExitCode}
& 'C:/Users/SuYanPei/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe' -NoProfile -File "$root/tools/launch_v15_finalizer.ps1"
exit $LASTEXITCODE
