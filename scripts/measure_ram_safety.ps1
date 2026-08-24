# Same sampling as measure_ram.ps1 (process working-set/private + system-wide
# free RAM), PLUS a safety cutoff: if system free RAM drops below
# -SafetyThresholdMB, terminate the monitored process immediately (don't wait
# for Windows to reach real memory-pressure crash territory, as happened in
# the original "Intento 1" APPCRASH), log an ABORT marker line, and exit.
# Used by scripts/run_sequence_campaign.py for the long-sequence campaign.
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [string]$OutCsv = "ram_report.csv",
    [int]$IntervalSeconds = 1,
    [int]$MaxSeconds = 3600,
    [int]$SafetyThresholdMB = 300
)

"timestamp,elapsed_s,proc_workingset_mb,proc_private_mb,sys_free_mb,sys_total_mb" |
    Out-File -FilePath $OutCsv -Encoding utf8

$start = Get-Date
while ($true) {
    $elapsed = ((Get-Date) - $start).TotalSeconds
    if ($elapsed -gt $MaxSeconds) {
        "ABORT_REASON=max_seconds_exceeded elapsed=$elapsed" | Out-File -FilePath "$OutCsv.abort" -Encoding utf8
        break
    }

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { break }

    $os = Get-CimInstance Win32_OperatingSystem
    $wsMb = [math]::Round($proc.WorkingSet64 / 1MB, 1)
    $privMb = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
    $freeMb = [math]::Round($os.FreePhysicalMemory / 1KB, 1)
    $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1KB, 1)

    "$(Get-Date -Format o),$([math]::Round($elapsed,1)),$wsMb,$privMb,$freeMb,$totalMb" |
        Out-File -FilePath $OutCsv -Append -Encoding utf8

    if ($freeMb -lt $SafetyThresholdMB) {
        "ABORT_REASON=safety_threshold_breached free_mb=$freeMb threshold_mb=$SafetyThresholdMB elapsed=$elapsed" |
            Out-File -FilePath "$OutCsv.abort" -Encoding utf8
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
