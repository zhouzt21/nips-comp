# Recovery

First scoring attempt stopped with KeyError `brier_mean`: the historical E3 per-benchmark JSON stores only ALC, not budget curves. Corrected the replay assertion to compare per-benchmark ALC at the same 1e-7 tolerance. No method, parameter, split or promotion rule changed; no candidate summary had been produced. A separate exploratory Python -c inspection failed because Windows native quoting removed inner quotes; inspection was completed with PowerShell ConvertFrom-Json instead.
