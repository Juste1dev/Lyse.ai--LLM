Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python .\scripts\daily_pipeline.py --config .\configs\daily_67m.yaml --phase auto
