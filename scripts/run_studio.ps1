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
$webCommand = "Set-Location '$Web'; npm run dev"

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $studioApiCommand)
Write-Host "Waiting for Studio API on 8000..." -ForegroundColor Yellow
if (-not (Wait-ForUrl "http://127.0.0.1:8000/docs" 30)) {
    throw "Studio API did not start on port 8000."
}

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $inferenceApiCommand)
Write-Host "Waiting for inference API on 8001..." -ForegroundColor Yellow
if (-not (Wait-ForUrl "http://127.0.0.1:8001/health" 90)) {
    throw "Inference API did not start on port 8001. Check the inference PowerShell window for the startup error."
}

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $webCommand)
Write-Host "Waiting for Studio Web on 3000..." -ForegroundColor Yellow
if (-not (Wait-ForUrl "http://127.0.0.1:3000" 60)) {
    throw "Studio Web did not start on port 3000. Check the Next.js PowerShell window."
}

Write-Host ""
Write-Host "AnomaVision Studio is ready." -ForegroundColor Green
Write-Host "Studio API:     http://localhost:8000" -ForegroundColor Green
Write-Host "Inference API:  http://localhost:8001" -ForegroundColor Green
Write-Host "Studio Web:     http://localhost:3000" -ForegroundColor Green

Start-Process "http://localhost:3000"
