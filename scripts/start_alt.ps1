$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

if (-not ("AltProcessJob" -as [type])) {
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class AltProcessJob
{
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetInformationJobObject(
        IntPtr hJob,
        int jobObjectInfoClass,
        IntPtr lpJobObjectInfo,
        uint cbJobObjectInfoLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr hObject);

    public const int JobObjectExtendedLimitInformation = 9;
    public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
}

public static class AltConsoleControl
{
    public delegate bool HandlerRoutine(uint ctrlType);

    private static HandlerRoutine _handler;
    private static int _appPid;
    private static int _workerPid;

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetConsoleCtrlHandler(HandlerRoutine handler, bool add);

    public static void Register(int appPid, int workerPid)
    {
        _appPid = appPid;
        _workerPid = workerPid;

        if (_handler == null)
        {
            _handler = new HandlerRoutine(HandleControlSignal);
        }

        SetConsoleCtrlHandler(_handler, true);
    }

    private static bool HandleControlSignal(uint ctrlType)
    {
        KillIfRunning(_appPid);
        KillIfRunning(_workerPid);
        return false;
    }

    private static void KillIfRunning(int pid)
    {
        if (pid <= 0)
        {
            return;
        }

        try
        {
            var process = System.Diagnostics.Process.GetProcessById(pid);
            if (!process.HasExited)
            {
                process.Kill();
            }
        }
        catch
        {
        }
    }
}
"@
}

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
        Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.Name -match '^(python.*|py|powershell|pwsh)\.exe$' -and
            $_.CommandLine -like "*$Needle*"
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

function New-KillOnCloseJob {
    $jobHandle = [AltProcessJob]::CreateJobObject([IntPtr]::Zero, $null)
    if ($jobHandle -eq [IntPtr]::Zero) {
        throw "Failed to create process job."
    }

    $info = New-Object AltProcessJob+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    $info.BasicLimitInformation.LimitFlags = [AltProcessJob]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    $infoLength = [System.Runtime.InteropServices.Marshal]::SizeOf($info)
    $infoPtr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($infoLength)
    try {
        [System.Runtime.InteropServices.Marshal]::StructureToPtr($info, $infoPtr, $false)
        $ok = [AltProcessJob]::SetInformationJobObject(
            $jobHandle,
            [AltProcessJob]::JobObjectExtendedLimitInformation,
            $infoPtr,
            [uint32]$infoLength
        )
        if (-not $ok) {
            throw "Failed to configure process job."
        }
    } finally {
        [System.Runtime.InteropServices.Marshal]::FreeHGlobal($infoPtr)
    }

    return $jobHandle
}

function Add-ProcessToJob {
    param(
        [IntPtr]$JobHandle,
        [System.Diagnostics.Process]$Process
    )

    if (-not [AltProcessJob]::AssignProcessToJobObject($JobHandle, $Process.Handle)) {
        throw "Failed to add process to job: PID=$($Process.Id)"
    }
}

function Start-ParentWatchdog {
    param(
        [string]$PythonExecutable,
        [int]$ParentPid,
        [int[]]$ChildPids
    )

    $watchdogRunner = $PythonExecutable
    if ($PythonExecutable -match "python(?:\.exe)?$") {
        $pythonwCandidate = Join-Path (Split-Path $PythonExecutable -Parent) "pythonw.exe"
        if (Test-Path $pythonwCandidate) {
            $watchdogRunner = $pythonwCandidate
        }
    }

    $watchdogLogPath = Join-Path $workspace ".runtime\logs\launcher.watchdog.log"
    Remove-Item -Path $watchdogLogPath -Force -ErrorAction SilentlyContinue

    $watchdogScript = @"
import ctypes
import subprocess
import sys
import time

log_path = r"$($watchdogLogPath -replace '\\', '\\')"
parent_pid = int(sys.argv[2])
child_pids = [int(value) for value in sys.argv[3:]]

SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102
kernel32 = ctypes.windll.kernel32


def log(message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def process_alive(pid: int) -> bool:
    handle = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


log(f"start parent={parent_pid} children={child_pids}")

while True:
    if not process_alive(parent_pid):
        log("parent_dead")
        for child_pid in child_pids:
            subprocess.run(
                ["taskkill", "/PID", str(child_pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log(f"killed={child_pid}")
        break

    if not any(process_alive(child_pid) for child_pid in child_pids):
        log("children_gone")
        break

    time.sleep(0.75)

log("exit")
"@

    Start-Process `
        -FilePath $watchdogRunner `
        -ArgumentList (@("-c", $watchdogScript, "alt-parent-watchdog", "$ParentPid") + ($ChildPids | ForEach-Object { "$_" })) `
        -WindowStyle Hidden `
        -PassThru | Out-Null
}

$jobHandle = New-KillOnCloseJob

$runner = Get-PythonRunner

$logDir = Join-Path $workspace ".runtime\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$streamlitNeedle = "-m streamlit run app.py"
$betaNeedle = "beta_server.py"
$workerNeedle = "-m jobs.scheduler"

# 기존 프로세스 종료 = 진짜 restart
Stop-ManagedProcess -Needle $workerNeedle
Stop-ManagedProcess -Needle $streamlitNeedle
Stop-ManagedProcess -Needle $betaNeedle

$workerArgs = $runner.PrefixArgs + @(
    "-m", "jobs.scheduler"
)

$appArgs = $runner.PrefixArgs + @(
    "beta_server.py",
    "--host", "0.0.0.0",
    "--port", "8505"
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

try {
    Add-ProcessToJob -JobHandle $jobHandle -Process $workerProc
    Add-ProcessToJob -JobHandle $jobHandle -Process $appProc
    [AltConsoleControl]::Register($appProc.Id, $workerProc.Id)
    Start-ParentWatchdog -PythonExecutable $runner.FilePath -ParentPid $PID -ChildPids @($appProc.Id, $workerProc.Id)

    Start-Sleep -Seconds 3

    Write-Host "Alt beta server/worker restart completed."
    Write-Host "Beta Server PID: $($appProc.Id)"
    Write-Host "Worker PID: $($workerProc.Id)"
    Write-Host "Logs: $logDir"
    Write-Host "Monitor URL: http://127.0.0.1:8505/beta"
    Write-Host "Keep this window open while the app is running."
    Write-Host "Closing this console will stop both app and worker."

    while ($true) {
        $appProc.Refresh()
        $workerProc.Refresh()
        if ($appProc.HasExited -or $workerProc.HasExited) {
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $appProc.HasExited) {
        Stop-Process -Id $appProc.Id -Force -ErrorAction SilentlyContinue
    }
    if (-not $workerProc.HasExited) {
        Stop-Process -Id $workerProc.Id -Force -ErrorAction SilentlyContinue
    }

    if ($appProc.HasExited) {
        Write-Warning "App process exited. ExitCode=$($appProc.ExitCode)"
    }
    if ($workerProc.HasExited) {
        Write-Warning "Worker process exited. ExitCode=$($workerProc.ExitCode)"
    }
} finally {
    if ($jobHandle -ne [IntPtr]::Zero) {
        [AltProcessJob]::CloseHandle($jobHandle) | Out-Null
    }
}
