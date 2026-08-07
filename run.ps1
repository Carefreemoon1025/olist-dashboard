# PowerShell 快速启动脚本
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

& python (Join-Path $PSScriptRoot "scripts\build_warehouse.py")
& streamlit run (Join-Path $PSScriptRoot "app.py")
