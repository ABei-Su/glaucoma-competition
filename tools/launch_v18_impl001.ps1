$ErrorActionPreference='Stop'
$root='E:/bisai/glaucoma_project'
trap { $_ | Out-String | Add-Content -LiteralPath "$root/reports/v18_impl001_watcher.stderr"; exit 1 }
Set-Location $root
if(Test-Path output/v18/implementation_001/finalization) {throw 'Preserve existing finalization'}
if(Test-Path 'E:/bisai/第十八版试交_v18') {throw 'Preserve existing delivery'}
$review=Get-Content reports/v18_impl001_review.json -Raw | ConvertFrom-Json
if(-not $review.approved){throw 'Independent review required'}
foreach($f in $review.files){if((Get-FileHash -LiteralPath $f.path -Algorithm SHA256).Hash.ToLower() -ne $f.sha256){throw 'Reviewed file changed'}}
if((Get-FileHash output/v18/implementation_001/frozen_inputs.json).Hash.ToLower() -ne $review.remediation_manifest_sha256){throw 'Review binding mismatch'}
$freeze=Get-Content output/v18/implementation_001/frozen_inputs.json -Raw|ConvertFrom-Json
foreach($f in $freeze.files){if((Get-FileHash -LiteralPath $f.path -Algorithm SHA256).Hash.ToLower() -ne $f.sha256){throw 'Frozen file changed'}}
$reservation=[IO.File]::Open("$root/output/v18/implementation_001/launch.reservation",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$b=[Text.Encoding]::UTF8.GetBytes([string]$PID);$reservation.Write($b,0,$b.Length)}finally{$reservation.Dispose()}
$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES='0'
$env:TEMP="$root/tmp";$env:TMP=$env:TEMP;$env:TORCH_HOME="$root/pretrain_model";$env:HF_HOME="$root/tmp/hf";$env:XDG_CACHE_HOME="$root/tmp/cache"
$p=Start-Process -FilePath 'C:/Users/SuYanPei/AppData/Local/Programs/Python/Python39/python.exe' -ArgumentList @('-u','-m','tools.finalize_v18_impl001') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/reports/v18_impl001.log" -RedirectStandardError "$root/reports/v18_impl001.stderr"
$null=$p.Handle
[pscustomobject]@{pid=$p.Id;watcher_pid=$PID;started_at=(Get-Date).ToString('o');deadline=(Get-Date).AddHours(4).ToString('o')} | ConvertTo-Json | Set-Content reports/v18_impl001_launch.json -Encoding utf8
if(-not $p.WaitForExit(14400000)) { & taskkill /PID $p.Id /T /F; [pscustomobject]@{outcome='budget_exceeded';time=(Get-Date).ToString('o');kill_exit=$LASTEXITCODE} | ConvertTo-Json | Set-Content reports/v18_impl001_exit.json -Encoding utf8;exit 1 }
$p.Refresh()
if($null -eq $p.ExitCode){throw 'Real exit unavailable; cannot claim success'}
[pscustomobject]@{outcome='exited';exit_code=$p.ExitCode;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content reports/v18_impl001_exit.json -Encoding utf8
exit $p.ExitCode
