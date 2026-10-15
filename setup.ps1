<# Crea/aggiorna soltanto l'ambiente DLA. Non esegue gli esercizi. #>
[CmdletBinding()]
param([string]$CondaBasePython)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

# Il launcher conda.bat di alcune installazioni Windows è inutilizzabile;
# il Python di base può invocare conda senza aggiornare base.
if (-not $CondaBasePython) {
    $candidates = @(
        (Join-Path $env:USERPROFILE 'miniconda3\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\python.exe')
    )
    $CondaBasePython = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $CondaBasePython -or -not (Test-Path -LiteralPath $CondaBasePython)) {
    throw 'Specificare -CondaBasePython con il percorso di python.exe dell''installazione Conda.'
}
$CondaBasePython = (Resolve-Path -LiteralPath $CondaBasePython).Path

function Invoke-DlaConda {
    param([string[]]$CondaArguments)
    # conda run attiva l'ambiente tramite un .bat; anche quel processo deve
    # invocare python -m conda, non il launcher conda.exe.
    $savedVariables = @{}
    $overrides = @{ CONDA_EXE = $CondaBasePython; _CONDA_EXE = $CondaBasePython; _CE_M = '-m'; _CE_CONDA = 'conda'; CONDA_CHANNELS = 'conda-forge' }
    foreach ($variable in $overrides.Keys) {
        $savedVariables[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
        [Environment]::SetEnvironmentVariable($variable, $overrides[$variable], 'Process')
    }
    try {
        & $CondaBasePython -m conda @CondaArguments
        if ($LASTEXITCODE -ne 0) { throw "Conda ha restituito $LASTEXITCODE" }
    } finally {
        foreach ($variable in $savedVariables.Keys) {
            [Environment]::SetEnvironmentVariable($variable, $savedVariables[$variable], 'Process')
        }
    }
}

Push-Location -LiteralPath $projectRoot
try {
    $envJson = & $CondaBasePython -m conda env list --json
    if ($LASTEXITCODE -ne 0) { throw 'Impossibile leggere gli ambienti Conda' }
    $environments = ($envJson -join "`n" | ConvertFrom-Json).envs
    $existing = @($environments | Where-Object { (Split-Path -Leaf $_) -ceq 'DLA' })
    if ($existing.Count -gt 1) { throw 'Più ambienti DLA trovati: risolvere l''ambiguità in Conda prima del setup' }
    if ($existing.Count -eq 1) {
        $nativeSpecs = @(Get-Content -LiteralPath 'environment.yml' | ForEach-Object {
            if ($_ -match '^\s+-\s+((?:python|pip)=\S+)\s*$') { $Matches[1] }
        })
        Invoke-DlaConda -CondaArguments (@('install', '--name', 'DLA', '--override-channels', '--channel', 'conda-forge', '--yes') + $nativeSpecs)
    } else {
        Invoke-DlaConda -CondaArguments @('create', '--name', 'DLA', '--file', 'environment.yml', '--override-channels', '--channel', 'conda-forge', '--yes')
    }
    Invoke-DlaConda -CondaArguments @('run', '--no-capture-output', '--name', 'DLA', 'python', '-m', 'pip',
        'install', '--upgrade', '--only-binary=:all:', '--requirement', 'requirements-gpu.txt')
    Invoke-DlaConda -CondaArguments @('run', '--no-capture-output', '--name', 'DLA', 'python', '-m', 'pip',
        'install', '--upgrade', '--only-binary=:all:', '--requirement', 'requirements.txt')

    # Registra anche le dipendenze indirette effettivamente installate.
    $metadataFolder = Join-Path $projectRoot 'outputs\environment'
    New-Item -ItemType Directory -Path $metadataFolder -Force | Out-Null
    $frozen = Invoke-DlaConda -CondaArguments @('run', '--name', 'DLA', 'python', '-m', 'pip', 'freeze')
    $frozen | Set-Content -LiteralPath (Join-Path $metadataFolder 'pip-freeze.txt') -Encoding utf8
    $exported = & $CondaBasePython -m conda env export --name DLA
    if ($LASTEXITCODE -ne 0) { throw 'Impossibile esportare l''ambiente Conda' }
    $exported | Set-Content -LiteralPath (Join-Path $metadataFolder 'environment-resolved.yml') -Encoding utf8
    Write-Host 'Ambiente DLA preparato. Nessun training, valutazione o test eseguito.'
    Write-Host "Attivazione: . .\activate.ps1 -CondaBasePython '$CondaBasePython'"
} finally {
    Pop-Location
}
