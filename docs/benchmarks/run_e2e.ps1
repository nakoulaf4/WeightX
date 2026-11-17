$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_CACHE = "C:\Users\runbo\.cache\huggingface\hub"
$env:PYTHONPATH = "C:\Users\runbo\Downloads\WeightX\src;C:\Users\runbo\Downloads\WeightX"
$env:WEIGHTX_PYTHON = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$work = "C:\Users\runbo\AppData\Local\Temp\qwenwork\wx-bench"
$wxRust = "C:\Users\runbo\Downloads\WeightX\rust\target\release\wx.exe"
$wxPy = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\wx.exe"
$py = "C:\Users\runbo\Downloads\WeightX\.venv\Scripts\python.exe"
$out = Join-Path $work "results_e2e.jsonl"
if (Test-Path $out) { Remove-Item $out }

function TimeIt([string]$label, [scriptblock]$block, [int]$n) {
    $times = @()
    for ($i = 0; $i -lt $n; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $block | Out-Null
        $sw.Stop()
        $times += $sw.Elapsed.TotalMilliseconds
    }
    $mean = ($times | Measure-Object -Average).Average
    $min = ($times | Measure-Object -Minimum).Minimum
    $obj = @{ cmd = $label; mean_ms = $mean; min_ms = $min; iters = $n } | ConvertTo-Json -Compress
    Add-Content -Path $out -Value $obj
    Write-Host $obj
}

TimeIt "startup-rust-wx-help" { & $wxRust --help } 5
TimeIt "startup-python-wx-help" { & $wxPy --help } 5
TimeIt "e2e-demo-rust" { & $wxRust demo } 1
TimeIt "e2e-demo-python" { & $py "C:\Users\runbo\Downloads\WeightX\examples\mvp_end_to_end_demo.py" } 1
Write-Host "E2E STAGE DONE"