$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_CACHE = "C:\Users\runbo\.cache\huggingface\hub"
$env:PYTHONPATH = "C:\Users\runbo\Downloads\WeightX\src;C:\Users\runbo\Downloads\WeightX"
$py = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$script = "C:\Users\runbo\Downloads\WeightX\docs\benchmarks\bench_py.py"
$work = "C:\Users\runbo\Downloads\WeightX\docs\benchmarks"
$out = Join-Path $work "results_py_cpu.jsonl"
if (Test-Path $out) { Remove-Item $out }
$meta = Get-Content (Join-Path $work "meta.json") | ConvertFrom-Json
$st = $meta.smollm2.safetensors
$smolT = ($meta.smollm2.targets -join ",")
$miniT = ($meta.minillama.targets -join ",")

function Run([string]$label, [string[]]$argv) {
    Write-Host "=== $label ==="
    $line = & $py $script @argv 2>$null
    if ($line) { Add-Content -Path $out -Value $line; Write-Host $line.Substring(0, [Math]::Min(300, $line.Length)) }
    else { Write-Host "FAILED: $label" }
}

Run "forward-smol-64" @("forward", "smollm2", "64", "5", "cpu")
Run "forward-smol-128" @("forward", "smollm2", "128", "3", "cpu")
Run "forward-gpt2-32" @("forward", "tinygpt2", "32", "5", "cpu")
Run "grads-smol-64" @("grads", "smollm2", "64", "3", "cpu")
Run "eamit-smol" @("eamit", "smollm2", $meta.smollm2.facts_file, $smolT, "5")
Run "eamit-mini" @("eamit", "minillama", $meta.minillama.facts_file, $miniT, "5")
Run "toon" @("toon", $meta.payload, "20")
Run "plan-smol" @("plan", "smollm2", "10")
Run "wsio-smol" @("wsio", $st, "3")
Run "describe-smol" @("describe", "smollm2", "model.layers.13.mlp.down_proj.weight", "10")
Run "sandbox-mini" @("sandbox", "minillama", "model.layers.1.mlp.down_proj.weight", "3")
Run "l2-smol-32" @("l2", "smollm2", "32", "cpu")
Write-Host "PY CPU STAGE DONE"