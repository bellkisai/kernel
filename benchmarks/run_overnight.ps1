#!/usr/bin/env pwsh
# ShrimPK Overnight Benchmark Orchestrator
#
# Runs the FULL benchmark suite in sequence:
#   Phase 1: Rust benchmarks (embedding-only, seeded, abstention, NR, multi-hop, expanded)
#   Phase 2: Rust consolidation benchmark (Ollama required)
#   Phase 3: LongMemEval-S validation (50 questions)
#   Phase 4: LongMemEval-S full (500 questions)
#
# Results logged to: benchmarks/results/overnight_YYYYMMDD_HHMM.jsonl
# Obsidian note updated: ShrimPK Kernel/Benchmark Results.md
# README updated if any benchmark hits 100%.
#
# Usage:
#   .\benchmarks\run_overnight.ps1
#   .\benchmarks\run_overnight.ps1 -SkipLongMemEval   (Rust benchmarks only)
#   .\benchmarks\run_overnight.ps1 -LongMemEvalLimit 50  (quick LongMemEval validation)

param(
    [switch]$SkipLongMemEval,
    [int]$LongMemEvalLimit = 0,  # 0 = full 500
    [string]$LongMemEvalModel = "gemma3:1b",
    [string]$RepoRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$ObsidianNote = (Join-Path $env:USERPROFILE "Obsidian Vault\ShrimPK Kernel\Benchmark Results.md")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$runStart = Get-Date
$dateStamp = $runStart.ToString("yyyyMMdd_HHmm")
$resultsDir = "$RepoRoot\benchmarks\results"
$resultsFile = "$resultsDir\overnight_$dateStamp.jsonl"
$logFile = "$resultsDir\overnight_$dateStamp.log"

if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

function Log {
    param([string]$msg)
    $ts = (Get-Date).ToString("HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

function AppendResult {
    param([hashtable]$record)
    $json = $record | ConvertTo-Json -Compress
    Add-Content -Path $resultsFile -Value $json -Encoding utf8
    Log "  RESULT: $json"
}

Log "============================================================"
Log "ShrimPK Overnight Benchmark Run — $dateStamp"
Log "============================================================"

Set-Location $RepoRoot

$commit = (& git rev-parse --short HEAD 2>$null)
$branch = (& git branch --show-current 2>$null)
Log "Branch: $branch  Commit: $commit"

# ===========================================================================
# Phase 1: Rust benchmarks (no Ollama)
# ===========================================================================
Log ""
Log "=== PHASE 1: Rust benchmarks (embedding-only / seeded) ==="

$rustTests = @(
    @{ Name = "embedding_only";   Filter = "micro_benchmark_embedding_only";   Label = "Embedding-only (20q)" },
    @{ Name = "seeded_children";  Filter = "benchmark_with_seeded_children";   Label = "Seeded children (20q, 100% target)" },
    @{ Name = "abstention";       Filter = "benchmark_abstention";             Label = "Abstention (5q absent facts)" },
    @{ Name = "negative_recall";  Filter = "benchmark_negative_recall";        Label = "Negative recall (3q superseded)" },
    @{ Name = "multi_hop";        Filter = "benchmark_multi_hop";              Label = "Multi-hop (4q 2-hop chain)" },
    @{ Name = "expanded_suite";   Filter = "benchmark_expanded_suite";         Label = "Expanded suite (20+5+3+4 = 32q)" }
)

foreach ($test in $rustTests) {
    Log ""
    Log "--- Running: $($test.Label) ---"
    $tStart = Get-Date

    $output = & cargo test --test echo_micro_benchmark -- --ignored --nocapture $($test.Filter) 2>&1
    $exitCode = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $tStart).TotalSeconds)

    # Print all output
    $output | ForEach-Object { Log "  $_" }

    # Parse results
    $passed = 0; $total = 0
    foreach ($line in $output) {
        if ($line -match "(?:MICRO-BENCHMARK|ABSTENTION BENCHMARK|NEGATIVE RECALL BENCHMARK|MULTI-HOP BENCHMARK|EXPANDED SUITE)\s+(?:RESULT|COMPLETE)?:?\s*(\d+)/(\d+)") {
            $passed = [int]$Matches[1]; $total = [int]$Matches[2]
        }
        # Catch "With seeded children: 20/20" format
        if ($line -match "(?:seeded children|embedding|consolidation):\s+(\d+)/(\d+)") {
            if ($total -eq 0) { $passed = [int]$Matches[1]; $total = [int]$Matches[2] }
        }
    }

    $pct = if ($total -gt 0) { [math]::Round($passed / $total * 100, 1) } else { 0 }
    Log "  SCORE: $passed/$total ($pct%) in ${elapsed}s — exit=$exitCode"

    AppendResult @{
        timestamp = (Get-Date -Format "o")
        phase     = "rust"
        commit    = $commit
        branch    = $branch
        test      = $test.Name
        label     = $test.Label
        passed    = $passed
        total     = $total
        pct       = $pct
        elapsed_s = $elapsed
        exit_code = $exitCode
    }

    if ($pct -ge 100.0 -and $total -gt 0) {
        Log "  *** 100% HIT — will update README ***"
    }
}

# ===========================================================================
# Phase 2: Rust consolidation benchmark (Ollama)
# ===========================================================================
Log ""
Log "=== PHASE 2: Consolidation benchmark (requires Ollama) ==="

$ollamaOk = $false
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaOk = $true
    Log "Ollama running — models: $(($tags.models | ForEach-Object { $_.name }) -join ', ')"
} catch {
    Log "WARNING: Ollama not reachable — skipping consolidation benchmark"
}

if ($ollamaOk) {
    Log "--- Running: Consolidation (20q, requires Ollama qwen2.5:1.5b) ---"
    $tStart = Get-Date
    $output = & cargo test --release --test echo_micro_benchmark -- --ignored --nocapture micro_benchmark_with_consolidation 2>&1
    $exitCode = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $tStart).TotalSeconds)

    $output | ForEach-Object { Log "  $_" }

    $passed = 0; $total = 0
    foreach ($line in $output) {
        if ($line -match "MICRO-BENCHMARK RESULT:\s+(\d+)/(\d+)") {
            $passed = [int]$Matches[1]; $total = [int]$Matches[2]
        }
        if ($line -match "With consolidation:\s+(\d+)/(\d+)") {
            if ($total -eq 0) { $passed = [int]$Matches[1]; $total = [int]$Matches[2] }
        }
    }
    $pct = if ($total -gt 0) { [math]::Round($passed / $total * 100, 1) } else { 0 }
    Log "  SCORE: $passed/$total ($pct%) in ${elapsed}s"

    AppendResult @{
        timestamp = (Get-Date -Format "o")
        phase     = "rust"
        commit    = $commit
        branch    = $branch
        test      = "consolidation"
        label     = "Consolidation (20q, qwen2.5:1.5b)"
        passed    = $passed
        total     = $total
        pct       = $pct
        elapsed_s = $elapsed
        exit_code = $exitCode
    }
}

# ===========================================================================
# Phase 3 & 4: LongMemEval
# ===========================================================================
if (-not $SkipLongMemEval) {
    Log ""
    Log "=== PHASE 3 & 4: LongMemEval-S ==="

    # Ensure daemon is running
    $daemonOk = $false
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:11435/health" -TimeoutSec 3
        $daemonOk = $true
        Log "ShrimPK daemon already running."
    } catch {
        Log "Starting ShrimPK daemon..."
        $daemonBin = "$RepoRoot\target\release\shrimpk-daemon.exe"
        if (Test-Path $daemonBin) {
            Start-Process -FilePath $daemonBin -WindowStyle Hidden
            $deadline = (Get-Date).AddSeconds(20)
            while ((Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
                try {
                    Invoke-RestMethod -Uri "http://127.0.0.1:11435/health" -TimeoutSec 2 | Out-Null
                    $daemonOk = $true
                    Log "Daemon started."
                    break
                } catch { }
            }
        } else {
            Log "WARNING: shrimpk-daemon.exe not found at $daemonBin"
        }
    }

    if (-not $ollamaOk) {
        Log "WARNING: Ollama not running — LongMemEval requires Ollama. Skipping."
    } elseif (-not $daemonOk) {
        Log "WARNING: Daemon not running — LongMemEval requires daemon. Skipping."
    } else {
        $datasetPath = "$RepoRoot\benchmarks\LongMemEval\data\longmemeval_s_cleaned.json"
        if (-not (Test-Path $datasetPath)) {
            Log "WARNING: Dataset not found at $datasetPath — skipping LongMemEval"
        } else {
            # Phase 3: validation run (50 questions)
            $validLimit = if ($LongMemEvalLimit -gt 0) { $LongMemEvalLimit } else { 50 }
            Log "--- Phase 3: LongMemEval validation ($validLimit questions, model=$LongMemEvalModel) ---"
            $outputPath = "$resultsDir\lme_validate_$dateStamp.jsonl"
            $tStart = Get-Date
            & python "$RepoRoot\benchmarks\run_longmemeval_v2.py" `
                --dataset $datasetPath `
                --output $outputPath `
                --model $LongMemEvalModel `
                --limit $validLimit `
                2>&1 | ForEach-Object { Log "  $_" }
            $elapsed = [math]::Round(((Get-Date) - $tStart).TotalSeconds)
            Log "  LongMemEval validation done in ${elapsed}s — output: $outputPath"

            AppendResult @{
                timestamp  = (Get-Date -Format "o")
                phase      = "longmemeval"
                commit     = $commit
                branch     = $branch
                test       = "longmemeval_s_validate"
                label      = "LongMemEval-S validation ($validLimit q)"
                model      = $LongMemEvalModel
                output     = $outputPath
                elapsed_s  = $elapsed
            }

            # Phase 4: Full 500-question run (only if not limited)
            if ($LongMemEvalLimit -eq 0) {
                Log ""
                Log "--- Phase 4: LongMemEval-S full (500 questions) ---"
                $fullOutputPath = "$resultsDir\lme_full_$dateStamp.jsonl"
                $tStart = Get-Date
                & python "$RepoRoot\benchmarks\run_longmemeval_v2.py" `
                    --dataset $datasetPath `
                    --output $fullOutputPath `
                    --model $LongMemEvalModel `
                    --resume `
                    2>&1 | ForEach-Object { Log "  $_" }
                $elapsed = [math]::Round(((Get-Date) - $tStart).TotalSeconds)
                Log "  LongMemEval full done in ${elapsed}s — output: $fullOutputPath"

                AppendResult @{
                    timestamp  = (Get-Date -Format "o")
                    phase      = "longmemeval"
                    commit     = $commit
                    branch     = $branch
                    test       = "longmemeval_s_full"
                    label      = "LongMemEval-S full (500q)"
                    model      = $LongMemEvalModel
                    output     = $fullOutputPath
                    elapsed_s  = $elapsed
                }
            }
        }
    }
}

# ===========================================================================
# Summary
# ===========================================================================
$totalElapsed = [math]::Round(((Get-Date) - $runStart).TotalSeconds / 60, 1)
Log ""
Log "============================================================"
Log "OVERNIGHT RUN COMPLETE — ${totalElapsed}min total"
Log "Results: $resultsFile"
Log "Log: $logFile"
Log "============================================================"

# Print all results in a table
Log ""
Log "RESULTS SUMMARY:"
$allResults = Get-Content $resultsFile | ForEach-Object { $_ | ConvertFrom-Json }
foreach ($r in $allResults) {
    if ($r.passed -ne $null) {
        $pctStr = "$($r.pct)%"
        Log "  $($r.label.PadRight(45)) $($r.passed)/$($r.total) ($pctStr)"
    } else {
        Log "  $($r.label.PadRight(45)) [output file: $(Split-Path $r.output -Leaf)]"
    }
}

# Append summary to Obsidian
$obsidianDir = Split-Path $ObsidianNote
if (-not (Test-Path $obsidianDir)) {
    New-Item -ItemType Directory -Path $obsidianDir -Force | Out-Null
}

$sessionHeader = "## Overnight Run — $dateStamp (commit $commit)"
$sessionRows = @("| Benchmark | Score | % | Phase |", "|-----------|-------|---|-------|")
foreach ($r in $allResults) {
    if ($r.passed -ne $null) {
        $sessionRows += "| $($r.label) | $($r.passed)/$($r.total) | $($r.pct)% | $($r.phase) |"
    }
}
$sessionContent = @($sessionHeader) + $sessionRows
$sessionContent | Add-Content -Path $ObsidianNote -Encoding utf8

Log "Obsidian note updated: $ObsidianNote"
Log "Done."
