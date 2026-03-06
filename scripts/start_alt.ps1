$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

function Get-PythonRunner {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            FilePath   = $python.Source
            PrefixArgs = @()
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{
            FilePath   = $py.Source
            PrefixArgs = @("-3")
        }
    }

    throw "python 실행 파일을 찾지 못했습니다."
}

function Find-ManagedProcess {
    param(
        [string]$Needle
    )

    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'py.exe' OR Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -like "*$Needle*"
        }
    )
}

function Stop-ManagedProcess {
    param(
        [string]$Needle
    )

    $procs = Find-ManagedProcess -Needle $Needle
    foreach ($proc in $procs) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Warning "프로세스 종료 실패: PID=$($proc.ProcessId)"
        }
    }

    if ($procs.Count -gt 0) {
        Start-Sleep -Seconds 2
    }
}

function Start-ManagedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$StdOutPath,
        [string]$StdErrPath,
        [switch]$Hidden
    )

    $params = @{
        FilePath               = $FilePath
        ArgumentList           = $ArgumentList
        WorkingDirectory       = $WorkingDirectory
        RedirectStandardOutput = $StdOutPath
        RedirectStandardError  = $StdErrPath
        PassThru               = $true
    }

    if ($Hidden) {
        $params.WindowStyle = "Hidden"
    }

    return Start-Process @params
}

$runner = Get-PythonRunner

$logDir = Join-Path $workspace ".runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$streamlitNeedle = "-m streamlit run app.py"
$workerNeedle = "-m jobs.scheduler"

# 기존 프로세스 종료 = 진짜 restart
Stop-ManagedProcess -Needle $workerNeedle
Stop-ManagedProcess -Needle $streamlitNeedle

$workerArgs = $runner.PrefixArgs + @(
    "-m", "jobs.scheduler"
)

$appArgs = $runner.PrefixArgs + @(
    "-m", "streamlit", "run", "app.py",
    "--server.address", "0.0.0.0",
    "--server.port", "8501",
    "--server.headless", "true"
)

$workerProc = Start-ManagedProcess `
    -FilePath $runner.FilePath `
    -ArgumentList $workerArgs `
    -WorkingDirectory $workspace `
    -StdOutPath (Join-Path $logDir "worker.stdout.log") `
    -StdErrPath (Join-Path $logDir "worker.stderr.log") `
    -Hidden

$appProc = Start-ManagedProcess `
    -FilePath $runner.FilePath `
    -ArgumentList $appArgs `
    -WorkingDirectory $workspace `
    -StdOutPath (Join-Path $logDir "app.stdout.log") `
    -StdErrPath (Join-Path $logDir "app.stderr.log") `
    -Hidden

Start-Sleep -Seconds 3

Write-Host "Alt app/worker restart completed."
Write-Host "App PID: $($appProc.Id)"
Write-Host "Worker PID: $($workerProc.Id)"
Write-Host "Logs: $logDir"