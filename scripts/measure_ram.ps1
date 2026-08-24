# Samples process working-set + system-wide free RAM at a fixed interval while a
# target PID is alive. Used to capture RAM peaks during demo.py / benchmark runs on
# this RAM-constrained machine (see CLAUDE.md — "Restricción de hardware crítica").
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [string]$OutCsv = "ram_report.csv",
    [int]$IntervalSeconds = 3,
    [int]$MaxSeconds = 1800
)

"timestamp,elapsed_s,proc_workingset_mb,proc_private_mb,sys_free_mb,sys_total_mb" |
    Out-File -FilePath $OutCsv -Encoding utf8

$start = Get-Date
while ($true) {
    $elapsed = ((Get-Date) - $start).TotalSeconds
    if ($elapsed -gt $MaxSeconds) { break }

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { break }

    $os = Get-CimInstance Win32_OperatingSystem
    $wsMb = [math]::Round($proc.WorkingSet64 / 1MB, 1)
    $privMb = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
    $freeMb = [math]::Round($os.FreePhysicalMemory / 1KB, 1)
    $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1KB, 1)

    "$(Get-Date -Format o),$([math]::Round($elapsed,1)),$wsMb,$privMb,$freeMb,$totalMb" |
        Out-File -FilePath $OutCsv -Append -Encoding utf8

    Start-Sleep -Seconds $IntervalSeconds
}
