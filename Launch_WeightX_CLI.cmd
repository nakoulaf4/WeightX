@echo off
rem WeightX hybrid — interactive CLI (Rust core + PyTorch sidecar)
chcp 65001 >nul
cd /d "C:\Users\runbo\Downloads\WeightX"
"C:\Users\runbo\Downloads\WeightX\dist\cli\wx.exe" chat %*
