"""Record non-secret runtime metadata used by plan_2."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from mlp.training_mechanism_data import write_json
from mlp.training_mechanism_guard import ROOT, sha

packages = ['torch','numpy','pandas','scipy','transformers','tokenizers','safetensors',
            'huggingface-hub','threadpoolctl','matplotlib','pyarrow']
hardware = json.loads(subprocess.check_output(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',
    '$c=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; '
    '@{cpu=$c.Name; cores=$c.NumberOfCores; logical_processors=$c.NumberOfLogicalProcessors; '
    'memory_bytes=$m.TotalPhysicalMemory} | ConvertTo-Json'],text=True))
runtime = Path('D:/nips-comp/paiec-runtime-win64-py312-cpu')
data = {'recorded_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
        'executable':sys.executable,'platform':platform.platform(),'hardware':hardware,
        'packages':{p:importlib.metadata.version(p) for p in packages},'torch_threads':4,
        'runtime_files':{str(runtime/name):sha(runtime/name) for name in
                         ('python.ps1','requirements-win-cpu.lock','requirements-external.lock')},
        'training_seeds':{'mlp':2026,'R43':4307},'fixed_episode_seed':73,
        'resampled_episode_seeds':list(range(5501,5521)),'cpu_only':True}
write_json(ROOT/'environment.json',data)
print(json.dumps(data,indent=2))
