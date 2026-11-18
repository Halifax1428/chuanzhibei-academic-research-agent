$ErrorActionPreference = 'Stop'

$AgentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$LangchainDirectory = Split-Path -Parent $AgentDirectory
$RepositoryDirectory = Split-Path -Parent $LangchainDirectory
$PythonExecutable = Join-Path $RepositoryDirectory '.venv\Scripts\python.exe'
$ServerUrl = 'http://127.0.0.1:8000/'
$HealthUrl = 'http://127.0.0.1:8000/api/health'

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Project virtual environment not found: $PythonExecutable. Install dependencies first."
}

$Listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $Listener) {
    $LogDirectory = Join-Path $AgentDirectory 'runtime_data\logs'
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $StdoutLog = Join-Path $LogDirectory 'server.stdout.log'
    $StderrLog = Join-Path $LogDirectory 'server.stderr.log'
    Start-Process -FilePath $PythonExecutable `
        -ArgumentList @('-m', 'uvicorn', 'agent.main:app', '--host', '127.0.0.1', '--port', '8000') `
        -WorkingDirectory $LangchainDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog | Out-Null
}

$Healthy = $false
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    try {
        $Health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        if ($Health.status -eq 'ok') {
            $Healthy = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $Healthy) {
    $ErrorLog = Join-Path $AgentDirectory 'runtime_data\logs\server.stderr.log'
    if (Test-Path -LiteralPath $ErrorLog) {
        Get-Content -LiteralPath $ErrorLog -Tail 30
    }
    throw 'Research assistant failed to start. Check the error log above.'
}

Start-Process $ServerUrl
Write-Host "Research assistant started: $ServerUrl"
Write-Host "Documents: $($Health.documents); model API configured: $($Health.model_configured)"
