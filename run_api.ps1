$repo = $PSScriptRoot
$venvPython = Join-Path (Split-Path $repo) ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found at: $venvPython"
}

Set-Location $repo
& $venvPython -m uvicorn api.main:app --host 0.0.0.0 --port 8000
