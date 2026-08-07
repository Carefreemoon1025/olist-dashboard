# PowerShell 快速启动脚本
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

if (-not (Test-Path (Join-Path $PSScriptRoot "data\warehouse\olist.duckdb"))) {
    & python (Join-Path $PSScriptRoot "scripts\generate_demo_data.py")
    & python (Join-Path $PSScriptRoot "scripts\build_warehouse.py")
}

& streamlit run (Join-Path $PSScriptRoot "app.py")
