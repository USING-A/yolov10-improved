$ErrorActionPreference = 'Continue'
$py = 'D:/Anaconda/envs/yolov10/python.exe'
$log = 'records/screening_to150_queue.log'

# Keep baseline + single-module runs on the same 150-epoch budget.
$jobs = @(
    @{ Name='baseline_bauto_nbs128'; Model='ultralytics/cfg/models/v10/yolov10n.yaml' },
    @{ Name='screen_yolov10n_edge_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_edge.yaml' },
    @{ Name='screen_yolov10n_cbam_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_cbam.yaml' },
    @{ Name='screen_yolov10n_eca_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_eca.yaml' },
    @{ Name='screen_yolov10n_bifpn_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_bifpn.yaml' }
)

function WriteLog([string]$msg) {
    Add-Content -Path $log -Value ((Get-Date).ToString('s') + ' ' + $msg)
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

function Get-RunActive([string]$runName) {
    $p = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match [Regex]::Escape($runName)
    }
    return (@($p).Count -gt 0)
}

function Get-ActiveRunCount {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and (
            $_.CommandLine -match 'baseline_bauto_nbs128' -or
            $_.CommandLine -match 'screen_yolov10n_edge_e100_auto' -or
            $_.CommandLine -match 'screen_yolov10n_cbam_e100_auto' -or
            $_.CommandLine -match 'screen_yolov10n_eca_e100_auto' -or
            $_.CommandLine -match 'screen_yolov10n_bifpn_e100_auto'
        )
    }
    return @($active).Count
}

WriteLog 'screening_to150 queue watcher started'

while ($true) {
    $pending = @()
    foreach ($j in $jobs) {
        $ep = Get-LastEpoch $j.Name
        if ($ep -lt 150) { $pending += $j }
    }

    if (-not $pending -or $pending.Count -eq 0) {
        WriteLog 'all target runs reached epoch>=150, watcher exits'
        break
    }

    $activeCount = Get-ActiveRunCount
    if ($activeCount -ge 2) {
        WriteLog "active_count=$activeCount (>=2), waiting"
        Start-Sleep -Seconds 60
        continue
    }

    foreach ($job in $pending) {
        if ((Get-ActiveRunCount) -ge 2) { break }
        if (Get-RunActive $job.Name) { continue }

        $runName = $job.Name
        $model = $job.Model
        $current = Get-LastEpoch $runName

        if ($current -ge 150) { continue }

        WriteLog "launching/resuming $runName from epoch=$current to target=150"
        $code = @"
import os
import sys
sys.path.insert(0, r'd:/Github Code/yolov10-improved')
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from ultralytics import YOLOv10

m = YOLOv10('$model')
m.train(
    data='datasets/apple_dataset2510_plus/apple1.yaml',
    cfg='configs/apple_hyp.yaml',
    epochs=150,
    workers=0,
    batch=-1,
    project='runs/apple',
    name='$runName',
    exist_ok=True,
    resume=True,
    plots=False,
    iou_type='ciou',
    cls_loss='bce'
)
"@

        $tmpPy = "records/${runName}_to150_launcher.py"
        $outLog = "records/${runName}_to150.out.log"
        $errLog = "records/${runName}_to150.err.log"
        try {
            Set-Content -Path $tmpPy -Value $code -Encoding UTF8
            Start-Process -FilePath $py -ArgumentList @($tmpPy) -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
            Start-Sleep -Seconds 5
            if (Get-RunActive $runName) {
                WriteLog "$runName launch confirmed"
            } else {
                WriteLog "$runName launch attempted but process not detected"
            }
        } catch {
            WriteLog "failed to launch ${runName}: $($_.Exception.Message)"
        }
    }

    Start-Sleep -Seconds 60
}

WriteLog 'screening_to150 queue watcher finished'
