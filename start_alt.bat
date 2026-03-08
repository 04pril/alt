@echo off
cd /d "%~dp0"

powershell -NoExit -NoProfile -ExecutionPolicy Bypass -Command ^
  "& '%~dp0scripts\start_alt.ps1';" ^
  "if ($?) {" ^
  "  $logDir = Join-Path '%~dp0' '.runtime\logs';" ^
  "  $files = @('app.stdout.log', 'app.stderr.log', 'worker.stdout.log', 'worker.stderr.log') | ForEach-Object { Join-Path $logDir $_ };" ^
  "  Write-Host '';" ^
  "  Write-Host 'Tailing Alt runtime logs. Press Ctrl+C to stop tailing. The app and worker keep running.' -ForegroundColor Yellow;" ^
  "  Write-Host ('Log directory: ' + $logDir) -ForegroundColor DarkGray;" ^
  "  Get-Content -Tail 20 -Wait $files" ^
  "}"
