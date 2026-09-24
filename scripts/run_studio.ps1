$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "apps\studio-web"

Write-Host "Starting AnomaVision Studio..." -ForegroundColor Cyan

# Studio workspace API
$studioApiCommand = "Set-Location '$Root'; uv run uvicorn apps.studio.api.app:app --host 127.0.0.1 --port 8000"

# Existing AnomaVision inference API
$inferenceApiCommand = "Set-Location '$Root'; $env:PORT=8001; uv run python api.py"

# Studio web application
$webCommand = "Set-Location '$Web'; npm run dev"

Start-Process powershell.exe -ArgumentList "-NoExit","-Command",$studioApiCommand
Start-Sleep -Seconds 2

Start-Process powershell.exe -ArgumentList "-NoExit","-Command",$inferenceApiCommand
Start-Sleep -Seconds 3

Start-Process powershell.exe -ArgumentList "-NoExit","-Command",$webCommand

Write-Host "Studio API:     http://localhost:8000" -ForegroundColor Green
Write-Host "Inference API:  http://localhost:8001" -ForegroundColor Green
Write-Host "Studio Web:     http://localhost:3000" -ForegroundColor Green
Write-Host "Wait a few seconds for Next.js, then open http://localhost:3000" -ForegroundColor Yellow

Start-Sleep -Seconds 4
Start-Process "http://localhost:3000"
