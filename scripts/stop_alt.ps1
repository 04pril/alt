$ErrorActionPreference = "SilentlyContinue"

$patterns = @(
    "-m streamlit run app.py",
    "beta_server.py",
    "-m jobs.scheduler"
)

Get-CimInstance Win32_Process | Where-Object {
    $commandLine = $_.CommandLine
    $commandLine -and @($patterns | Where-Object { $commandLine -like "*$_*" }).Count -gt 0
} | ForEach-Object {
    try {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
    } catch {
    }
}

Write-Host "Alt server/worker stop requested."
