$ErrorActionPreference='Stop'
$root='E:/bisai/glaucoma_project'
trap { $_ | Out-String | Add-Content -LiteralPath "$root/reports/v16_watcher.stderr"; exit 1 }
Set-Location $root
if(Test-Path runs/v16_r50_refine4) {throw 'Preserve existing run'}
if(Test-Path reports/v16_launch.json) {throw 'Duplicate launch forbidden'}
$review=Get-Content reports/v16_preflight_review.json -Raw | ConvertFrom-Json
if(-not $review.approved) {throw 'Independent review required'}
$freeze=Get-Content output/v16/frozen_inputs.json -Raw | ConvertFrom-Json
foreach($entry in $freeze.files) {if((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) {throw "Frozen input changed: $($entry.path)"}}
# Atomic reservation persists as audit evidence; no automatic cleanup/retry.
$reservation=[IO.File]::Open("$root/output/v16/launch.reservation",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
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
$p=Start-Process -FilePath $python -ArgumentList @('-u','train.py','--architecture','resnet50','--cache','data/cache384','--run','runs/v16_r50_refine4','--epochs','4','--checkpoint-epoch','4','--size','384','--batch','16','--workers','4','--seed','20261033','--lr','0.00001','--init-checkpoint','runs/v15_full_r50_8/best.pt','--pos-weight','1','--augmentation','mild','--validation-probability-precision','fp32') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root/reports/v16_train.log" -RedirectStandardError "$root/reports/v16_train.stderr" -PassThru
[pscustomobject]@{pid=$p.Id;watcher_pid=$PID;started_at=(Get-Date).ToString('o');deadline=(Get-Date).AddHours(3).ToString('o');plan='docs/science-superpowers/plans/2026-10-30-v16-refinement.md'} | ConvertTo-Json | Set-Content reports/v16_launch.json -Encoding UTF8
$finished=$p.WaitForExit(10800000)
if(-not $finished) {
 & taskkill /PID $p.Id /T /F
 [pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='budget_exceeded';pid=$p.Id;kill_exit=$LASTEXITCODE} | ConvertTo-Json | Set-Content reports/v16_exit.json -Encoding UTF8
 exit 1
}
$p.Refresh()
[pscustomobject]@{finished_at=(Get-Date).ToString('o');outcome='exited';pid=$p.Id;exit_code=$p.ExitCode} | ConvertTo-Json | Set-Content reports/v16_exit.json -Encoding UTF8
if($p.ExitCode -ne 0) {exit $p.ExitCode}
& 'C:/Users/SuYanPei/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe' -NoProfile -File "$root/tools/launch_v16_finalizer.ps1"
exit $LASTEXITCODE
