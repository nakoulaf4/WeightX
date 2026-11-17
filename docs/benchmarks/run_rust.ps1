$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:WEIGHTX_PYTHON = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$bench = "C:\Users\runbo\Downloads\WeightX\rust\target\release\wx-bench.exe"
$work = "C:\Users\runbo\Downloads\WeightX\docs\benchmarks"
$out = Join-Path $work "results_rust.jsonl"
if (Test-Path $out) { Remove-Item $out }
$smol = "HuggingFaceTB/SmolLM2-135M"
$gpt2 = "sshleifer/tiny-gpt2"
$mini = "minillama-demo"
$st = (Get-Content (Join-Path $work "meta.json") | ConvertFrom-Json).smollm2.safetensors

function Run([string]$label, [string[]]$argv) {
    Write-Host "=== $label ==="
    $line = & $bench @argv 2>$null
    if ($line) { Add-Content -Path $out -Value $line; Write-Host $line.Substring(0, [Math]::Min(300, $line.Length)) }
    else { Write-Host "FAILED: $label" }
}

Run "forward-smol-64" @("forward", $smol, "64", "5")
Run "forward-smol-128" @("forward", $smol, "128", "3")
Run "forward-gpt2-32" @("forward", $gpt2, "32", "5")
Run "grads-smol-64" @("grads", $smol, "64", "3")
Run "eamit-smol" @("eamit", $smol, (Join-Path $work "facts_smollm2.json"), "model.layers.13.mlp.down_proj.weight,model.layers.17.mlp.down_proj.weight,model.layers.22.mlp.down_proj.weight,model.layers.26.mlp.down_proj.weight", "5")
Run "eamit-mini" @("eamit", $mini, (Join-Path $work "facts_minillama.json"), "model.layers.1.mlp.down_proj.weight,model.layers.2.mlp.down_proj.weight,model.layers.3.mlp.down_proj.weight", "5")
Run "toon" @("toon", (Join-Path $work "payload.json"), "20")
Run "plan-smol" @("plan", $smol, "10")
Run "wsio-smol" @("wsio", $st, "3")
Run "describe-smol" @("describe", $smol, "model.layers.13.mlp.down_proj.weight", "10")
Run "sandbox-mini" @("sandbox", $mini, "model.layers.1.mlp.down_proj.weight", "3")
Run "l2-smol-32" @("l2", $smol, "32")
Write-Host "RUST STAGE DONE"