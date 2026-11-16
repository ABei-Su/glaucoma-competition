$ErrorActionPreference='Stop'
Set-Location 'E:/bisai/glaucoma_project'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:TEMP='E:/bisai/glaucoma_project/tmp'
$env:TMP=$env:TEMP
$env:TORCH_HOME='E:/bisai/glaucoma_project/pretrain_model'
$env:CUDA_VISIBLE_DEVICES='0'
$py='C:/Users/SuYanPei/AppData/Local/Programs/Python/Python39/python.exe'
foreach($kind in @('control','mixup')) {
 $folder="E:/bisai/最新试交_v13_两份/$kind"
 if(Test-Path "$folder/submission.csv") { throw 'Existing CSV, refusing overwrite' }
 & $py -u predict.py --checkpoint "$folder/config.json" --output "$folder/submission.csv"
 if($LASTEXITCODE -ne 0) { throw "Prediction failed: $kind" }
}
