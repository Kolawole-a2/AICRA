param(
  [string]$TargetDir = $env:AICRA_EMBER2024_DIR
)

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
  $TargetDir = Join-Path (Get-Location) "data\ember2024_real"
}

Write-Host "AICRA EMBER-2024 data check"
Write-Host "Expected directory: $TargetDir"

if (Test-Path $TargetDir) {
  $sample = Get-ChildItem -Path $TargetDir -Filter "*_train.jsonl" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -ne $sample) {
    Write-Host "OK: Found dataset directory and a sample file:" $sample.FullName
    exit 0
  } else {
    Write-Host "WARNING: Directory exists but no '*_train.jsonl' files found."
    Write-Host "Place your EMBER-2024 JSONL split files here."
    exit 2
  }
} else {
  Write-Host "MISSING: EMBER-2024 directory not found."
  Write-Host ""
  Write-Host "To set up:"
  Write-Host "1) Obtain EMBER-2024 JSONL split files via your approved source."
  Write-Host "2) Create the directory: $TargetDir"
  Write-Host "3) Place the JSONL files inside it (example: '*_train.jsonl')."
  Write-Host ""
  Write-Host "Optional: set AICRA_EMBER2024_DIR to your dataset location."
  Write-Host "See docs/DATA.md for details."
  exit 1
}
