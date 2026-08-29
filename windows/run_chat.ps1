param(
    [string]$Checkpoint,
    [string]$Tokenizer = ".\artifacts\tokenizer.model"
)
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python .\scripts\chat_tui.py --checkpoint $Checkpoint --tokenizer $Tokenizer
