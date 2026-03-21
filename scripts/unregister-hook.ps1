# ShrimPK — Remove Claude Code hook from settings.json
# Called by the MSI installer during uninstall.

$settingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"

if (-not (Test-Path $settingsPath)) {
    exit 0
}

try {
    $raw = Get-Content $settingsPath -Raw -ErrorAction Stop
    if (-not $raw -or $raw.Trim().Length -eq 0) {
        exit 0
    }
    $settings = $raw | ConvertFrom-Json -ErrorAction Stop
} catch {
    exit 0
}

# Navigate to hooks.UserPromptSubmit
$hooks = $settings.hooks
if (-not $hooks) { exit 0 }

$ups = $hooks.UserPromptSubmit
if (-not $ups) { exit 0 }

# Filter out shrimpk hook entries
$filtered = @()
foreach ($entry in $ups) {
    $isShrimpk = $false
    $entryHooks = $entry.hooks
    if ($entryHooks) {
        foreach ($hk in $entryHooks) {
            if ($hk.command -and $hk.command -like "*shrimpk-hook*") {
                $isShrimpk = $true
                break
            }
        }
    }
    if (-not $isShrimpk) {
        $filtered += $entry
    }
}

$hooks.UserPromptSubmit = $filtered

# Write back (UTF-8 without BOM)
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
    $json = $settings | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($settingsPath, $json, $utf8NoBom)
}
