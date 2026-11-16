$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\bisai\glaucoma_project'
$pythonPath = 'C:\Users\SuYanPei\AppData\Local\Programs\Python\Python39\python.exe'
if (Test-Path -LiteralPath "$projectRoot\reports\v13_status.json") { throw 'Existing v13 state; no duplicate launch' }
if (Test-Path -LiteralPath "$projectRoot\output\v13") { throw 'Existing v13 output; preserve evidence' }
if (Test-Path -LiteralPath "$projectRoot\reports\v13_coordinator.log") { throw 'Existing launch log; inspect before recovery' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = "$projectRoot\tmp"
$env:TMP = "$projectRoot\tmp"
$env:TORCH_HOME = "$projectRoot\pretrain_model"
$env:HF_HOME = "$projectRoot\tmp\hf"
$env:XDG_CACHE_HOME = "$projectRoot\tmp\cache"
$env:CUDA_VISIBLE_DEVICES = '0'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','tools.run_v13') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput "$projectRoot\reports\v13_coordinator.log" -RedirectStandardError "$projectRoot\reports\v13_coordinator.stderr" -PassThru
[pscustomobject]@{pid=$process.Id; started_at=(Get-Date).ToString('o'); root=$projectRoot} | ConvertTo-Json | Set-Content -LiteralPath "$projectRoot\reports\v13_launch.json" -Encoding utf8
Get-Content -LiteralPath "$projectRoot\reports\v13_launch.json"
