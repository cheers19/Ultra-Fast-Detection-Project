# Kill stuck Jupyter / IPython kernels and clear stale connection files.
# Run while notebooks are CLOSED, then reload Cursor (Ctrl+Shift+P -> Developer: Reload Window).

$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping ipykernel / kernel_interrupt_daemon processes..."
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object {
        $_.CommandLine -match 'ipykernel_launcher|kernel_interrupt_daemon|jupyter'
    } |
    ForEach-Object {
        Write-Host "  kill PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }

Start-Sleep -Seconds 1

$rt = Join-Path $env:APPDATA "jupyter\runtime"
if (Test-Path $rt) {
    $n = (Get-ChildItem $rt -Filter "kernel-*.json").Count
    Remove-Item (Join-Path $rt "kernel-*.json") -Force
    Remove-Item (Join-Path $rt "kernel-*.log") -Force
    Write-Host "Removed $n stale files from $rt"
}

Write-Host "Done. In Cursor: close notebook tabs -> Reload Window -> Select Kernel: Ultra-Fast (.venv 3.12)"
