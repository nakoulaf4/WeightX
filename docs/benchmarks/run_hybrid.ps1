$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:WEIGHTX_PYTHON = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$env:WEIGHTX_BACKEND = "auto"
$bench = "C:\Users\runbo\Downloads\WeightX\rust\target\release\wx-bench.exe"
$work = "C:\Users\runbo\Downloads\WeightX\docs\benchmarks"
$out = Join-Path $work "results_hybrid.jsonl"
if (Test-Path $out) { Remove-Item $out }
$smol = "HuggingFaceTB/SmolLM2-135M"
$mini = "minillama-demo"
$targets = "model.layers.13.mlp.down_proj.weight,model.layers.17.mlp.down_proj.weight,model.layers.22.mlp.down_proj.weight,model.layers.26.mlp.down_proj.weight"
$miniTargets = "model.layers.1.mlp.down_proj.weight,model.layers.2.mlp.down_proj.weight,model.layers.3.mlp.down_proj.weight"

function Run([string]$label, [string[]]$argv) {
    Write-Host "=== $label ==="
    $line = & $bench @argv 2>$null
    if ($line) { Add-Content -Path $out -Value $line; Write-Host $line.Substring(0, [Math]::Min(300, $line.Length)) }
    else { Write-Host "FAILED: $label" }
}

$env:WEIGHTX_TORCH_DEVICE = "cpu"
Run "hybrid-forward-smol-64-cpu" @("forward", $smol, "64", "5")
Run "hybrid-grads-smol-64-cpu" @("grads", $smol, "64", "3")
$env:WEIGHTX_TORCH_DEVICE = "cuda"
Run "hybrid-forward-smol-64-cuda" @("forward", $smol, "64", "5")
Run "hybrid-forward-smol-128-cuda" @("forward", $smol, "128", "3")
Run "hybrid-grads-smol-64-cuda" @("grads", $smol, "64", "3")
Run "hybrid-eamit-smol" @("eamit", $smol, (Join-Path $work "facts_smollm2.json"), $targets, "3")
Run "hybrid-l2-smol-32-cuda" @("l2", $smol, "32")
$env:WEIGHTX_TORCH_DEVICE = "auto"
Run "hybrid-eamit-mini-stays-rust" @("eamit", $mini, (Join-Path $work "facts_minillama.json"), $miniTargets, "5")
Run "hybrid-plan-smol-stays-rust" @("plan", $smol, "10")
Write-Host "HYBRID STAGE DONE"