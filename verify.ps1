# infer-lab verification (Windows PowerShell wrapper)
#   .\verify.ps1
$ErrorActionPreference = "Stop"
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $python scripts\verify.py @args
exit $LASTEXITCODE
