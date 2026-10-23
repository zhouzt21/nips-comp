$ErrorActionPreference = 'Stop'
$outPath = 'D:\nips-comp\nips-comp\paiec_baseline\runs\research_loop\R76\process_memory.csv'
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'shared_slope_outer.py' }
if (@($targets).Count -ne 1) { throw 'Expected exactly one R76 Python process' }
$taskPid = $targets.ProcessId
$samples = New-Object System.Collections.Generic.List[object]
while ($true) {
    $proc = Get-Process -Id $taskPid -ErrorAction SilentlyContinue
    if ($null -eq $proc) { break }
    $samples.Add([PSCustomObject]@{
        utc = (Get-Date).ToUniversalTime().ToString('o')
        pid = $taskPid
        working_set_mb = $proc.WorkingSet64 / 1MB
        lifetime_peak_working_set_mb = $proc.PeakWorkingSet64 / 1MB
        private_memory_mb = $proc.PrivateMemorySize64 / 1MB
        cpu_seconds = $proc.CPU
    })
    $samples | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $outPath
    Start-Sleep -Seconds 5
}
Write-Output ('R76 memory samples: ' + $samples.Count)
