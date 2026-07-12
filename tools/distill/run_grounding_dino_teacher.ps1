param(
    [Parameter(Mandatory = $true)] [string] $DataRoot,
    [Parameter(Mandatory = $true)] [string] $AnnotationRoot,
    [Parameter(Mandatory = $true)] [string] $WorkDir,
    [int] $Epochs = 50,
    [int] $SmokeSamples = 0,
    [string] $Scale = "800,1333",
    [int] $BatchSize = 1,
    [int] $AccumulationSteps = 16,
    [int] $NumWorkers = 2
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mmdetRoot = "D:\Github Code\.tmp\mmdetection-v3.3.0"
$python = "D:\Anaconda\envs\groundingdino\python.exe"
if (-not (Test-Path -LiteralPath $mmdetRoot)) { throw "MMDetection checkout is missing: $mmdetRoot" }
if (-not (Test-Path -LiteralPath $python)) { throw "Teacher Python is missing: $python" }

$env:MMDET_ROOT = $mmdetRoot
$env:PYTHONUTF8 = "1"
$env:GDINO_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
$env:GDINO_ANNOTATION_ROOT = (Resolve-Path -LiteralPath $AnnotationRoot).Path
$env:GDINO_MAX_EPOCHS = "$Epochs"
$env:GDINO_BATCH_SIZE = "$BatchSize"
$env:GDINO_ACCUM_STEPS = "$AccumulationSteps"
$env:GDINO_NUM_WORKERS = "$NumWorkers"
$env:GDINO_SCALE = $Scale
$env:GDINO_SMOKE_SAMPLES = "$SmokeSamples"

$config = Join-Path $repoRoot "configs\distill\grounding_dino_swin_t_teacher.py"
& $python (Join-Path $mmdetRoot "tools\train.py") $config --work-dir $WorkDir
exit $LASTEXITCODE
