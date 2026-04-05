$ErrorActionPreference = 'Continue'
$py = 'D:/Anaconda/envs/yolov10/python.exe'
$log = 'records/step9_queue.log'
$model = 'ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml'
$data = 'datasets/apple_dataset2510_plus/apple1.yaml'
$hyp = 'configs/apple_hyp.yaml'

$jobs = @(
    @{
        Name = 'tune_cbam_eca_strategy_cosine_ls_e60'
        Epochs = 60
        Extra = "iou_type='ciou', cls_loss='bce', cos_lr=True, label_smoothing=0.05, close_mosaic=20"
    },
    @{
        Name = 'tune_cbam_eca_loss_giou_varifocal_e60'
        Epochs = 60
        Extra = "iou_type='giou', cls_loss='varifocal', focal_gamma=2.0, focal_alpha=0.75"
    }
)

function Write-Log([string]$msg) {
    Add-Content -Path $log -Value ((Get-Date).ToString('s') + ' ' + $msg)
}

function Get-ActiveTuneCount {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'tune_cbam_eca_'
    }
    return @($active).Count
}

function GetRunActive([string]$runName) {
    $p = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match [Regex]::Escape($runName)
    }
    return (@($p).Count -gt 0)
}

function Get-LastEpoch([string]$runName) {
    $csv = "runs/apple/$runName/results.csv"
    if (-not (Test-Path $csv)) { return -1 }
    try {
        $rows = Import-Csv $csv
        if (-not $rows -or $rows.Count -eq 0) { return -1 }
        return [int]($rows | Select-Object -Last 1).epoch
    } catch {
        return -1
    }
}

function InvokeQueueRun($job) {
    $name = $job.Name
    $epochs = [int]$job.Epochs
    $last = Test-Path "runs/apple/$name/weights/last.pt"
    $resume = if ($last) { 'True' } else { 'False' }

    $code = @"
import os, sys
sys.path.insert(0, r'd:/Github Code/yolov10-improved')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from ultralytics import YOLOv10
m = YOLOv10('$model')
m.train(data='$data', cfg='$hyp', epochs=$epochs, workers=0, batch=-1, project='runs/apple', name='$name', exist_ok=True, resume=$resume, plots=False, $($job.Extra))
"@

    Write-Log "launching $name resume=$resume"
    Start-Process -FilePath $py -ArgumentList @('-c', $code) -WindowStyle Hidden
}

Write-Log 'step9 queue scheduler started'

$pending = New-Object System.Collections.ArrayList
$jobs | ForEach-Object { [void]$pending.Add($_) }

while ($pending.Count -gt 0) {
    for ($i = $pending.Count - 1; $i -ge 0; $i--) {
        $job = $pending[$i]
        $name = $job.Name
        $target = [int]$job.Epochs

        $ep = Get-LastEpoch $name
        if ($ep -ge $target) {
            Write-Log "$name already completed at epoch=$ep"
            $pending.RemoveAt($i)
            continue
        }

        if (GetRunActive $name) {
            continue
        }

        $active = Get-ActiveTuneCount
        if ($active -ge 2) {
            continue
        }

        InvokeQueueRun $job
        Start-Sleep -Seconds 5
    }

    Start-Sleep -Seconds 30
}

Write-Log 'step9 queue scheduler finished all jobs'
