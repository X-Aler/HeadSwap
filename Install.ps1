$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'
$env:UV_HTTP_TIMEOUT = '1800'
$uv = Join-Path $PSScriptRoot '.bootstrap\bin\uv.exe'
if (-not (Test-Path -LiteralPath $uv)) {
    New-Item -ItemType Directory -Path '.bootstrap' -Force | Out-Null
    $zip = Join-Path $PSScriptRoot '.bootstrap\uv.zip'
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.9.8/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath '.bootstrap\bin' -Force
}
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit code $LASTEXITCODE)." }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $uv venv --python 3.12 .venv
    Check-Exit 'Python installation'
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Write-Host 'Installing PyTorch with CUDA 12.8 (supports RTX 2060)...'
if (-not (Test-Path -LiteralPath '.venv\.torch-ready')) {
    & $python download_torch.py
    Check-Exit 'CUDA package download'
    & $uv --quiet pip install --python $python '.cache\wheels\torch-2.10.0+cu128-cp312-cp312-win_amd64.whl' --index-url 'https://download.pytorch.org/whl/cu128'
    Check-Exit 'PyTorch installation'
    New-Item -ItemType File -Path '.venv\.torch-ready' -Force | Out-Null
}
Write-Host 'Installing application libraries...'
& $uv --quiet pip install --python $python -r requirements.txt
Check-Exit 'Application dependencies'
New-Item -ItemType File -Path '.venv\.dependencies-ready' -Force | Out-Null
Write-Host 'Downloading models. About 5.2 GB; interrupted downloads can be resumed.'
& $python download_models.py
Check-Exit 'Model download'
Write-Host 'Ready. Double-click Start.cmd.' -ForegroundColor Green
