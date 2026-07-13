param(
    [Parameter(Mandatory = $true)] [string] $DataRoot,
    [Parameter(Mandatory = $true)] [string] $AnnotationRoot,
    [Parameter(Mandatory = $true)] [string] $WorkDir,
    [Parameter(Mandatory = $true)] [string] $LogPath,
    [int] $Epochs = 50,
    [string] $Scale = "800,1333",
    [int] $BatchSize = 1,
    [int] $AccumulationSteps = 16,
    [int] $NumWorkers = 2
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "run_grounding_dino_teacher.ps1"
& $launcher -DataRoot $DataRoot -AnnotationRoot $AnnotationRoot -WorkDir $WorkDir `
    -Epochs $Epochs -Scale $Scale -BatchSize $BatchSize `
    -AccumulationSteps $AccumulationSteps -NumWorkers $NumWorkers 2>&1 |
    Tee-Object -FilePath $LogPath
exit $LASTEXITCODE
