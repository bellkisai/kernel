# ShrimPK — Register Claude Code hook in settings.json
# Called by the MSI installer after file installation.
# Usage: powershell -ExecutionPolicy Bypass -File register-hook.ps1 -InstallDir "C:\...\ShrimPK"

param(
    [Parameter(Mandatory=$true)]
    [string]$InstallDir
)

$settingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"
$settingsDir = Split-Path $settingsPath

# Ensure .claude directory exists
if (-not (Test-Path $settingsDir)) {
    New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
}

# Read existing settings or start fresh
$settings = @{}
if (Test-Path $settingsPath) {
    try {
        $raw = Get-Content $settingsPath -Raw -ErrorAction Stop
        if ($raw -and $raw.Trim().Length -gt 0) {
            $settings = $raw | ConvertFrom-Json -ErrorAction Stop
            # Convert PSCustomObject to hashtable for easier manipulation
            $ht = @{}
            $settings.PSObject.Properties | ForEach-Object { $ht[$_.Name] = $_.Value }
            $settings = $ht
        }
    } catch {
        # If JSON is corrupt, preserve non-hooks keys by trying a repair
        # but fall back to empty if totally broken
        $settings = @{}
    }
}

# Build the hook entry
$hookScriptPath = ($InstallDir -replace '\\', '/') + "/scripts/shrimpk-hook.sh"
$hookCommand = "bash $hookScriptPath"

$hookEntry = @{
    matcher = ""
    hooks = @(
        @{
            type = "command"
            command = $hookCommand
            timeout = 10
        }
    )
}

# Ensure hooks.UserPromptSubmit exists
if (-not $settings.ContainsKey("hooks")) {
    $settings["hooks"] = @{}
}

$hooks = $settings["hooks"]

# Handle PSCustomObject from JSON parse
if ($hooks -is [System.Management.Automation.PSCustomObject]) {
    $hooksHt = @{}
    $hooks.PSObject.Properties | ForEach-Object { $hooksHt[$_.Name] = $_.Value }
    $hooks = $hooksHt
    $settings["hooks"] = $hooks
}

if (-not $hooks.ContainsKey("UserPromptSubmit")) {
    $hooks["UserPromptSubmit"] = @()
}

$ups = $hooks["UserPromptSubmit"]

# Convert to array if needed
if ($ups -isnot [System.Array]) {
    $ups = @($ups)
}

# Remove any existing shrimpk hook entries
$filtered = @()
foreach ($entry in $ups) {
    $isShrimpk = $false
    $entryHooks = $entry.hooks
    if (-not $entryHooks) {
        try { $entryHooks = $entry.PSObject.Properties["hooks"].Value } catch {}
    }
    if ($entryHooks) {
        foreach ($hk in $entryHooks) {
            $cmd = $hk.command
            if (-not $cmd) {
                try { $cmd = $hk.PSObject.Properties["command"].Value } catch {}
            }
            if ($cmd -and $cmd -like "*shrimpk-hook*") {
                $isShrimpk = $true
                break
            }
        }
    }
    if (-not $isShrimpk) {
        $filtered += $entry
    }
}

# Add our hook
$filtered += $hookEntry
$hooks["UserPromptSubmit"] = $filtered
$settings["hooks"] = $hooks

# Write back with proper JSON formatting (UTF-8 without BOM)
# Use Python for clean JSON output — PowerShell's ConvertTo-Json has excessive indentation
$jsonCompact = $settings | ConvertTo-Json -Depth 10 -Compress
$tempFile = [System.IO.Path]::GetTempFileName()
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($tempFile, $jsonCompact, $utf8NoBom)
$pretty = python3 -c "import json; d=json.load(open(r'$tempFile')); print(json.dumps(d,indent=2))" 2>$null
Remove-Item $tempFile -ErrorAction SilentlyContinue
if ($pretty) {
    $prettyStr = if ($pretty -is [System.Array]) { $pretty -join "`n" } else { $pretty }
    [System.IO.File]::WriteAllText($settingsPath, $prettyStr, $utf8NoBom)
} else {
    # Fallback: write PowerShell's formatting
    $json = $settings | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($settingsPath, $json, $utf8NoBom)
}
