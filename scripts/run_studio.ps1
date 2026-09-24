$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "apps\studio-web"

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    return $false
}

Write-Host "Starting AnomaVision Studio..." -ForegroundColor Cyan

$studioApiCommand = "Set-Location '$Root'; uv run uvicorn apps.studio.api.app:app --host 127.0.0.1 --port 8000"
$inferenceApiCommand = "Set-Location '$Root'; " + $envLine + " uv run python api.py"
$webCommand = "Set-Location '$Web'; $env:PORT='3000'; npm run dev"

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $studioApiCommand)
Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $inferenceApiCommand)
Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $webCommand)

Write-Host "Waiting for Studio Web on 3000..." -ForegroundColor Yellow
if (-not (Wait-ForUrl "http://127.0.0.1:3000" 90)) {
    throw "Studio Web did not start on port 3000. Check the Next.js PowerShell window."
}

Write-Host "Studio API: waiting on 8000..." -ForegroundColor Yellow
$studioReady = Wait-ForUrl "http://127.0.0.1:8000/docs" 60

Write-Host "Inference API: waiting on 8001..." -ForegroundColor Yellow
$inferenceReady = Wait-ForUrl "http://127.0.0.1:8001/health" 90

Write-Host ""
Write-Host "Studio Web:     http://localhost:3000" -ForegroundColor Green
if ($studioReady) {
    Write-Host "Studio API:     http://localhost:8000" -ForegroundColor Green
} else {
    Write-Host "Studio API:     NOT READY" -ForegroundColor Red
}
if ($inferenceReady) {
    Write-Host "Inference API:  http://localhost:8001" -ForegroundColor Green
} else {
    Write-Host "Inference API:  NOT READY" -ForegroundColor Red
}

Start-Process "http://localhost:3000"
