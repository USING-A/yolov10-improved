$ErrorActionPreference = 'Continue'
$py = 'D:/Anaconda/envs/yolov10/python.exe'
$jobs = @(
    @{ Name='screen_yolov10n_cbam_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_cbam.yaml' },
    @{ Name='screen_yolov10n_eca_e100_auto';  Model='ultralytics/cfg/models/v10/yolov10n_eca.yaml' },
    @{ Name='screen_yolov10n_bifpn_e100_auto'; Model='ultralytics/cfg/models/v10/yolov10n_bifpn.yaml' }
)
$log = 'records/module_scheduler.log'
Add-Content -Path $log -Value ("`n=== Scheduler v2 start " + (Get-Date) + " ===")
$lastLaunch = @{}

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

while ($true) {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'screen_yolov10n_(cbam|eca|bifpn)_e100_auto'
    }
    if ($active) {
        Add-Content -Path $log -Value ((Get-Date).ToString('s') + ' active module process detected, waiting...')
        Start-Sleep -Seconds 120
        continue
    }

    $next = $null
    foreach ($j in $jobs) {
        $ep = Get-LastEpoch $j.Name
        if ($ep -ge 100) { continue }
        $next = $j
        break
    }

    if (-not $next) {
        Add-Content -Path $log -Value ((Get-Date).ToString('s') + ' all module jobs completed.')
        break
    }

    $name = $next.Name
    $model = $next.Model

    # Cooldown guard: if a job failed fast, avoid relaunching it in a tight loop.
    if ($lastLaunch.ContainsKey($name)) {
        $elapsed = (Get-Date) - $lastLaunch[$name]
        if ($elapsed.TotalSeconds -lt 300) {
            Add-Content -Path $log -Value ((Get-Date).ToString('s') + " cooldown active for $name ($([int](300 - $elapsed.TotalSeconds))s left), waiting...")
            Start-Sleep -Seconds 60
            continue
        }
    }

    Add-Content -Path $log -Value ((Get-Date).ToString('s') + " launching $name with batch=-1")

    $code = @"
from ultralytics import YOLOv10
m = YOLOv10('$model')
m.train(data='datasets/apple_dataset2510_plus/apple1.yaml', cfg='configs/apple_hyp.yaml', epochs=100, workers=0, batch=-1, project='runs/apple', name='$name', exist_ok=True)
"@

    $outLog = "records/$name.log"
    $errLog = "records/$name.err.log"
    try {
        Start-Process -FilePath $py -ArgumentList @('-c', $code) -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
        $lastLaunch[$name] = Get-Date
    } catch {
        Add-Content -Path $log -Value ((Get-Date).ToString('s') + " failed to launch ${name}: $($_.Exception.Message)")
    }
    Start-Sleep -Seconds 30
}
