$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "apps\studio-web"

Write-Host "Starting AnomaVision Studio..." -ForegroundColor Cyan

$apiCommand = "Set-Location '$Root'; uv run uvicorn apps.studio.api.app:app --host 127.0.0.1 --port 8000"
$webCommand = "Set-Location '$Web'; npm run dev"

Start-Process powershell.exe -ArgumentList "-NoExit","-Command",$apiCommand
Start-Sleep -Seconds 2
Start-Process powershell.exe -ArgumentList "-NoExit","-Command",$webCommand

Write-Host "Studio API: http://localhost:8000" -ForegroundColor Green
Write-Host "Studio Web: http://localhost:3000" -ForegroundColor Green
Write-Host "Wait a few seconds for Next.js, then open http://localhost:3000" -ForegroundColor Yellow

Start-Sleep -Seconds 4
Start-Process "http://localhost:3000"
