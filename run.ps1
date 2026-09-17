# infer-lab stack runner (Windows PowerShell wrapper)
#   .\run.ps1
$ErrorActionPreference = "Stop"
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $python scripts\run_stack.py @args
exit $LASTEXITCODE
