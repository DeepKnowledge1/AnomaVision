$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Web = Join-Path $Root "apps\studio-web"
$LogDir = Join-Path $Root ".studio-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-PortProcess {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($connection in $connections) {
            if ($connection.OwningProcess -and $connection.OwningProcess -ne $PID) {
                Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    } catch { }
}

function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSeconds = 60, [int]$ExpectedStatus = 200)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($response.StatusCode -eq $ExpectedStatus) { return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}

Write-Host "Starting AnomaVision Studio..." -ForegroundColor Cyan
Stop-PortProcess 3000
Stop-PortProcess 8000
Stop-PortProcess 8001

$studioApiCommand = "Set-Location '$Root'; uv run uvicorn apps.studio.api.app:app --host 127.0.0.1 --port 8000"
$inferenceApiCommand = "Set-Location '$Root'; [Environment]::SetEnvironmentVariable('PORT','8001','Process'); uv run python apps\api.py"
$webCommand = "Set-Location '$Web'; [Environment]::SetEnvironmentVariable('PORT','3000','Process'); npm run dev"

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $studioApiCommand) -RedirectStandardOutput (Join-Path $LogDir "studio-api.log") -RedirectStandardError (Join-Path $LogDir "studio-api.err.log")
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $inferenceApiCommand) -RedirectStandardOutput (Join-Path $LogDir "inference-api.log") -RedirectStandardError (Join-Path $LogDir "inference-api.err.log")
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoProfile", "-Command", $webCommand) -RedirectStandardOutput (Join-Path $LogDir "web.log") -RedirectStandardError (Join-Path $LogDir "web.err.log")

Write-Host "Waiting for Studio Web on 3000..." -ForegroundColor Yellow
$webReady = Wait-ForUrl "http://127.0.0.1:3000" 120
Write-Host "Studio API: waiting on 8000..." -ForegroundColor Yellow
$studioReady = Wait-ForUrl "http://127.0.0.1:8000/docs" 90
Write-Host "Inference API: waiting on 8001..." -ForegroundColor Yellow
$inferenceReady = Wait-ForUrl "http://127.0.0.1:8001/health" 180

Write-Host ""
if ($webReady) { Write-Host "Studio Web:     http://localhost:3000  READY" -ForegroundColor Green } else { Write-Host "Studio Web:     NOT READY" -ForegroundColor Red }
if ($studioReady) { Write-Host "Studio API:     http://localhost:8000  READY" -ForegroundColor Green } else { Write-Host "Studio API:     NOT READY" -ForegroundColor Red }
if ($inferenceReady) {
    Write-Host "Inference API:  http://localhost:8001  READY" -ForegroundColor Green
} else {
    Write-Host "Inference API:  NOT READY" -ForegroundColor Red
    Write-Host "Check .studio-logs\inference-api.log and inference-api.err.log for details." -ForegroundColor Red
}

if ($webReady) { Start-Process "http://localhost:3000" } else { throw "Studio Web did not start on port 3000. Check the Next.js PowerShell window." }