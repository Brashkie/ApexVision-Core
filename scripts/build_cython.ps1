# ApexVision-Core — Build Cython extensions (Windows)
Write-Host "Building Cython extensions..." -ForegroundColor Cyan
Set-Location $PSScriptRoot/..
python python/setup.py build_ext --inplace
if ($LASTEXITCODE -eq 0) {
    Write-Host "Cython build OK" -ForegroundColor Green
} else {
    Write-Host "Cython build FAILED" -ForegroundColor Red
    exit 1
}
