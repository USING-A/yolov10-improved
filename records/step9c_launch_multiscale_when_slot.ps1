$ErrorActionPreference = 'Continue'
$log = 'records/step9c_queue.log'
$py = 'D:/Anaconda/envs/yolov10/python.exe'
$runName = 'tune_cbam_eca_strategy_multiscale_e60'

function WriteLog([string]$msg) {
    Add-Content -Path $log -Value ((Get-Date).ToString('s') + ' ' + $msg)
}

function GetActiveCount {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'tune_cbam_eca_'
    }
    return @($active).Count
}

function IsRunActive([string]$name) {
    $p = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match [Regex]::Escape($name)
    }
    return (@($p).Count -gt 0)
}

function GetLastEpoch([string]$name) {
    $csv = "runs/apple/$name/results.csv"
    if (-not (Test-Path $csv)) { return -1 }
    try {
        $rows = Import-Csv $csv
        if (-not $rows -or $rows.Count -eq 0) { return -1 }
        return [int]($rows | Select-Object -Last 1).epoch
    } catch {
        return -1
    }
}

WriteLog 'step9c single-run queue watcher started'

$ep = GetLastEpoch $runName
if ($ep -ge 60) {
    WriteLog "$runName already complete at epoch=$ep"
    exit 0
}

while ($true) {
    if (IsRunActive $runName) {
        WriteLog "$runName already active, watcher exits"
        break
    }

    $active = GetActiveCount
    if ($active -lt 2) {
        $code = @"
import os, sys
sys.path.insert(0, r'd:/Github Code/yolov10-improved')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from ultralytics import YOLOv10
m = YOLOv10('ultralytics/cfg/models/v10/yolov10n_cbam_eca.yaml')
m.train(data='datasets/apple_dataset2510_plus/apple1.yaml', cfg='configs/apple_hyp.yaml', epochs=60, workers=0, batch=-1, project='runs/apple', name='tune_cbam_eca_strategy_multiscale_e60', exist_ok=True, resume=False, plots=False, multi_scale=True, warmup_epochs=5.0, iou_type='ciou', cls_loss='bce')
"@
        WriteLog "launching $runName because active_count=$active"
        Start-Process -FilePath $py -ArgumentList @('-c', $code) -WindowStyle Hidden
        Start-Sleep -Seconds 3
        if (IsRunActive $runName) {
            WriteLog "$runName launch confirmed"
        } else {
            WriteLog "$runName launch attempted but process not detected"
        }
        break
    }

    Start-Sleep -Seconds 30
}

WriteLog 'step9c single-run queue watcher finished'
