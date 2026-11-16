$ErrorActionPreference='Stop'
$root='E:/bisai/glaucoma_project'
Set-Location $root
if(Test-Path output/v14/finalization) {throw 'Preserve existing finalization'}
$res=[IO.File]::Open("$root/output/v14/finalization.reservation",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try {$bytes=[Text.Encoding]::UTF8.GetBytes([string]$PID);$res.Write($bytes,0,$bytes.Length)} finally {$res.Dispose()}
$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:CUDA_VISIBLE_DEVICES='0'
$env:TEMP="$root/tmp";$env:TMP=$env:TEMP;$env:TORCH_HOME="$root/pretrain_model";$env:HF_HOME="$root/tmp/hf";$env:XDG_CACHE_HOME="$root/tmp/cache"
$p=Start-Process -FilePath 'C:/Users/SuYanPei/AppData/Local/Programs/Python/Python39/python.exe' -ArgumentList @('-u','-m','tools.finalize_v14') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/reports/v14_finalizer.log" -RedirectStandardError "$root/reports/v14_finalizer.stderr"
[pscustomobject]@{pid=$p.Id;watcher_pid=$PID;started_at=(Get-Date).ToString('o');deadline=(Get-Date).AddHours(4).ToString('o')} | ConvertTo-Json | Set-Content reports/v14_finalizer_launch.json -Encoding UTF8
if(-not $p.WaitForExit(14400000)) { & taskkill /PID $p.Id /T /F; [pscustomobject]@{outcome='budget_exceeded';time=(Get-Date).ToString('o');kill_exit=$LASTEXITCODE} | ConvertTo-Json | Set-Content reports/v14_finalizer_exit.json -Encoding UTF8;exit 1 }
$p.Refresh()
[pscustomobject]@{outcome='exited';exit_code=$p.ExitCode;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content reports/v14_finalizer_exit.json -Encoding UTF8
exit $p.ExitCode
