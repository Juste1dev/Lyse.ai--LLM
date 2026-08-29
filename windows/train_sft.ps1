param(
    [string]$BaseCheckpoint
)
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python .\scripts\train_sft.py --config .\configs\base_67m.yaml --base-checkpoint $BaseCheckpoint
