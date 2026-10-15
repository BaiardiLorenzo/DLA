<# Dot-source: . .\activate.ps1. Modifica soltanto la sessione PowerShell. #>
[CmdletBinding()]
param([string]$CondaBasePython)

if (-not $CondaBasePython) {
    $CondaBasePython = @(
        (Join-Path $env:USERPROFILE 'miniconda3\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\python.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $CondaBasePython -or -not (Test-Path -LiteralPath $CondaBasePython)) {
    throw 'Specificare -CondaBasePython con il Python di base di Conda.'
}
$CondaBasePython = (Resolve-Path -LiteralPath $CondaBasePython).Path
$condaRoot = Split-Path -Parent $CondaBasePython
$modulePath = Join-Path $condaRoot 'shell\condabin\Conda.psm1'
if (-not (Test-Path -LiteralPath $modulePath)) { throw 'Modulo PowerShell Conda non trovato' }

# Il modulo ufficiale usa queste variabili ad ogni chiamata.
$env:CONDA_EXE = $CondaBasePython
$env:_CONDA_EXE = $CondaBasePython
$env:_CE_M = '-m'
$env:_CE_CONDA = 'conda'
$env:CONDA_PYTHON_EXE = $CondaBasePython
$env:_CONDA_ROOT = $condaRoot
Import-Module $modulePath -ArgumentList @{ ChangePs1 = $true } -Global
conda activate DLA
if ($LASTEXITCODE -ne 0 -or $env:CONDA_DEFAULT_ENV -ne 'DLA') { throw 'Attivazione DLA fallita: eseguire prima setup.ps1' }

# L'attivazione rigenera anche i metadati del launcher: ripristina l'invocazione Python.
$env:CONDA_EXE = $CondaBasePython
$env:_CONDA_EXE = $CondaBasePython
$env:_CE_M = '-m'
$env:_CE_CONDA = 'conda'
Write-Host "DLA attivo: $env:CONDA_PREFIX"
