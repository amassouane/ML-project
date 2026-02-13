param(
    [string]$DataPath = 'data/raw/heart.csv',
    [string]$PythonExe = '.venv\\Scripts\\python.exe'
)

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe. Activate your venv or pass -PythonExe."
    exit 1
}

& $PythonExe src/run_pipeline.py --data-path $DataPath
