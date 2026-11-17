$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_CACHE = "C:\Users\runbo\.cache\huggingface\hub"
$env:PYTHONPATH = "C:\Users\runbo\Downloads\WeightX\src;C:\Users\runbo\Downloads\WeightX"
$py = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$script = "C:\Users\runbo\AppData\Local\Temp\qwenwork\wx-bench\bench_py.py"
$work = "C:\Users\runbo\AppData\Local\Temp\qwenwork\wx-bench"
$out = Join-Path $work "results_py_cuda.jsonl"
if (Test-Path $out) { Remove-Item $out }

function Run([string]$label, [string[]]$argv) {
    Write-Host "=== $label ==="
    $line = & $py $script @argv 2>$null
    if ($line) { Add-Content -Path $out -Value $line; Write-Host $line.Substring(0, [Math]::Min(300, $line.Length)) }
    else { Write-Host "FAILED: $label" }
}

Run "forward-smol-64-cuda" @("forward", "smollm2", "64", "10", "cuda")
Run "forward-smol-128-cuda" @("forward", "smollm2", "128", "5", "cuda")
Run "grads-smol-64-cuda" @("grads", "smollm2", "64", "5", "cuda")
Run "l2-smol-32-cuda" @("l2", "smollm2", "32", "cuda")
Write-Host "PY CUDA STAGE DONE"