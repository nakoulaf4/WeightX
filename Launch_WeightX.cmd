@echo off
rem WeightX hybrid — Tauri desktop (Rust core + PyTorch sidecar)
chcp 65001 >nul
cd /d "C:\Users\runbo\Downloads\WeightX"
start "" "C:\Users\runbo\Downloads\WeightX\dist\gui\WeightX.exe" %*
