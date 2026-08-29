Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python .\scripts\train.py --config .\configs\base_67m.yaml --resume auto
