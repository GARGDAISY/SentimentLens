param(
    [switch]$InstallLanguageModels
)

$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Make sure Python 3.12 is installed and available as 'py'." }

$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

if ($InstallLanguageModels) {
    Write-Host "Installing six Argos language-to-English models (~650 MB)..." -ForegroundColor Cyan
    & $Python scripts\install_argos_models.py
    if ($LASTEXITCODE -ne 0) { throw "One or more Argos models could not be installed. See messages above." }
} else {
    Write-Host "Dependencies installed. To add offline language models, run: .\.venv\Scripts\python.exe scripts\install_argos_models.py" -ForegroundColor Yellow
}

Write-Host "Setup complete. Start the app with:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\python.exe -m streamlit run app.py"
