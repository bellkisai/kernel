# ============================================================
# ShrimPK Installer for Windows
# Run: powershell -ExecutionPolicy Bypass -File scripts\install.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$INSTALL_DIR = "$env:USERPROFILE\.shrimpk\bin"
$KERNEL_DIR = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $KERNEL_DIR) { $KERNEL_DIR = (Get-Location).Path }

Write-Host ""
Write-Host "  🦐 ShrimPK Installer" -ForegroundColor Cyan
Write-Host "  =====================" -ForegroundColor Cyan
Write-Host "  Push-based AI memory where memories find you."
Write-Host ""

# ---- Step 1: Build ----
Write-Host "[1/6] Building release binaries..." -ForegroundColor Yellow
Push-Location $KERNEL_DIR
cargo build --release -p shrimpk-cli -p shrimpk-mcp -p shrimpk-daemon -p shrimpk-app 2>&1 | Select-Object -Last 3
Pop-Location
Write-Host ""

# ---- Step 2: Install to PATH ----
Write-Host "[2/6] Installing to $INSTALL_DIR..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $INSTALL_DIR | Out-Null

$binaries = @("shrimpk.exe", "shrimpk-mcp.exe", "shrimpk-daemon.exe", "shrimpk-app.exe")
foreach ($bin in $binaries) {
    $src = Join-Path "$KERNEL_DIR\target\release" $bin
    if (Test-Path $src) {
        Copy-Item $src $INSTALL_DIR -Force
        Write-Host "  Copied $bin"
    } else {
        Write-Host "  WARNING: $bin not found" -ForegroundColor Red
    }
}

# Add to Windows user PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*\.shrimpk\bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$INSTALL_DIR", "User")
    Write-Host "  Added to PATH: $INSTALL_DIR" -ForegroundColor Green
} else {
    Write-Host "  Already on PATH"
}
# Also set for current session
$env:Path = "$INSTALL_DIR;$env:Path"
Write-Host ""

# ---- Step 3: Register MCP ----
Write-Host "[3/6] Registering MCP server with Claude Code..." -ForegroundColor Yellow
$claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
if ($claudeCmd) {
    claude mcp remove shrimpk 2>$null
    claude mcp add --transport stdio --scope user shrimpk -- "$INSTALL_DIR\shrimpk-mcp.exe"
    Write-Host "  MCP server registered globally" -ForegroundColor Green
} else {
    Write-Host "  'claude' CLI not found. Register manually after installing Claude Code:" -ForegroundColor DarkYellow
    Write-Host "    claude mcp add --transport stdio --scope user shrimpk -- $INSTALL_DIR\shrimpk-mcp.exe"
}
Write-Host ""

# ---- Step 4: Start daemon ----
Write-Host "[4/6] Starting daemon..." -ForegroundColor Yellow
Stop-Process -Name "shrimpk-daemon" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Start-Process -FilePath "$INSTALL_DIR\shrimpk-daemon.exe" -WindowStyle Hidden
Write-Host "  Daemon starting on localhost:11435..."
Start-Sleep -Seconds 5

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:11435/health" -TimeoutSec 5
    Write-Host "  Daemon running ($($health.memories) memories loaded)" -ForegroundColor Green
} catch {
    Write-Host "  Daemon may still be loading (first run downloads 23MB model)" -ForegroundColor DarkYellow
    Write-Host "  Check: curl http://localhost:11435/health"
}
Write-Host ""

# ---- Step 5: Auto-start ----
Write-Host "[5/6] Installing auto-start on login..." -ForegroundColor Yellow
& "$INSTALL_DIR\shrimpk-daemon.exe" --install
Write-Host ""

# ---- Step 6: Verify ----
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow
Write-Host ""

# Check binary
if (Test-Path "$INSTALL_DIR\shrimpk.exe") {
    Write-Host "  ✓ shrimpk CLI installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ shrimpk CLI not found" -ForegroundColor Red
}

# Check daemon
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:11435/health" -TimeoutSec 2
    Write-Host "  ✓ Daemon running ($($health.memories) memories)" -ForegroundColor Green
} catch {
    Write-Host "  ⏳ Daemon starting..." -ForegroundColor DarkYellow
}

# Check MCP
if ($claudeCmd) {
    $mcpList = claude mcp list 2>&1
    if ($mcpList -match "shrimpk") {
        Write-Host "  ✓ MCP server registered" -ForegroundColor Green
    }
}

# Check PATH
if (Get-Command shrimpk -ErrorAction SilentlyContinue) {
    Write-Host "  ✓ shrimpk on PATH" -ForegroundColor Green
} else {
    Write-Host "  ⏳ Restart terminal for PATH to take effect" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host "  🦐 ShrimPK installed!" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Daemon:  http://localhost:11435"
Write-Host "  CLI:     shrimpk store / echo / stats / status"
Write-Host "  Tray:    shrimpk-app (🦐 in taskbar)"
Write-Host "  Data:    ~/.shrimpk-kernel/"
Write-Host ""
Write-Host "  Restart your terminal, then try:" -ForegroundColor White
Write-Host "    shrimpk status" -ForegroundColor White
Write-Host ""
