param(
    [string]$Python = (Join-Path $env:USERPROFILE 'miniconda3\envs\DLA\python.exe')
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$workerPath = Join-Path $projectRoot 'finish_remaining.py'
$statePath = Join-Path $projectRoot 'outputs\experiments_20261014\background_state.json'
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.status -eq 'running') {
        $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.pid)"
        if ($existingProcess -and $existingProcess.ExecutablePath -eq $pythonPath -and
            $existingProcess.CommandLine.Contains($workerPath)) {
            Write-Output "Coordinatore già attivo, PID $($state.pid). Nessun job duplicato."
            return
        }
    }
}
# WMI avvia il processo dal servizio di Windows, fuori dall'albero dei processi
# del terminale/Codex. Nessuna finestra e nessun monitoraggio tramite modelli.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0
}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = ('"{0}" "{1}"' -f $pythonPath, $workerPath)
    CurrentDirectory = $projectRoot
    ProcessStartupInformation = $startup
}
if ($created.ReturnValue -ne 0) {
    throw "Avvio background non riuscito, codice Windows $($created.ReturnValue)"
}
Write-Output "Coordinatore locale avviato, PID $($created.ProcessId)."
Write-Output "Report automatico: $(Join-Path $projectRoot 'RISULTATI.md')"
Write-Output "Log: $(Join-Path $projectRoot 'outputs\experiments_20261014\background.log')"
