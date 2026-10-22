# Launch only the bundled interpreter and the separately copied research deps.
# The user's environment is restored when this command finishes.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DepsPath = $env:PAIEC_RESEARCH_DEPS,
    [switch]$Online,
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)
$ErrorActionPreference = 'Stop'
$interpreter = Join-Path $PSScriptRoot 'runtime\python.exe'
if (-not (Test-Path -LiteralPath $interpreter -PathType Leaf)) {
    throw 'Run python.ps1 from the extracted runtime bundle (runtime/python.exe is missing).'
}
if (-not $DepsPath) { $DepsPath = 'D:\paiec_research_deps' }
if (-not (Test-Path -LiteralPath $DepsPath -PathType Container)) {
    throw 'Set PAIEC_RESEARCH_DEPS or pass -DepsPath to the research dependency folder you copied separately.'
}
$DepsPath = (Resolve-Path -LiteralPath $DepsPath).Path
foreach ($package in @('huggingface_hub', 'safetensors', 'tokenizers', 'transformers')) {
    if (-not (Test-Path -LiteralPath (Join-Path $DepsPath $package) -PathType Container)) {
        throw "The external dependency folder is missing $package. See requirements-external.lock."
    }
}
$git = Get-Command git -ErrorAction SilentlyContinue
$cleanPath = @((Join-Path $PSScriptRoot 'runtime'), (Join-Path $PSScriptRoot 'runtime\DLLs'),
    "$env:SystemRoot\System32", "$env:SystemRoot", "$env:SystemRoot\System32\Wbem")
if ($git) { $cleanPath += Split-Path -Parent $git.Source }
$settings = @{
    'PYTHONHOME' = $null
    'PYTHONPATH' = $DepsPath
    'PYTHONNOUSERSITE' = '1'
    'PYTHONDONTWRITEBYTECODE' = '1'
    'PYTHONUTF8' = '1'
    'PAIEC_RESEARCH_DEPS' = $DepsPath
    'HF_HOME' = (Join-Path $PSScriptRoot 'cache\huggingface')
    'HF_HUB_OFFLINE' = $(if ($Online) { '0' } else { '1' })
    'TRANSFORMERS_OFFLINE' = $(if ($Online) { '0' } else { '1' })
    'OPENBLAS_NUM_THREADS' = '1'
    'OMP_NUM_THREADS' = '4'
    'MKL_NUM_THREADS' = '4'
    'PATH' = ($cleanPath -join ';')
}
$previous = @{}
foreach ($name in $settings.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
}
try {
    & $interpreter -s @PythonArgs
    $result = $LASTEXITCODE
}
finally {
    foreach ($name in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
exit $result
