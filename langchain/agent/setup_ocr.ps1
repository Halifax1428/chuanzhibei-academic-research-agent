$ErrorActionPreference = 'Stop'

$LocalTesseract = Join-Path $env:LOCALAPPDATA 'Programs\Tesseract-OCR\tesseract.exe'
$MachineTesseract = Join-Path $env:ProgramFiles 'Tesseract-OCR\tesseract.exe'
$TesseractExecutable = @($LocalTesseract, $MachineTesseract) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $TesseractExecutable) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Install Tesseract OCR 5.x, then rerun this script.'
    }
    winget install --id tesseract-ocr.tesseract --exact --silent --accept-package-agreements --accept-source-agreements
    $TesseractExecutable = @($LocalTesseract, $MachineTesseract) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $TesseractExecutable) {
    throw 'Tesseract installation did not produce a tesseract.exe in a known install folder.'
}

$DataDirectory = Join-Path $env:LOCALAPPDATA 'Tesseract-OCR\tessdata'
New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null

foreach ($Language in @('eng', 'chi_sim')) {
    $Target = Join-Path $DataDirectory "$Language.traineddata"
    $MetadataUrl = "https://api.github.com/repos/tesseract-ocr/tessdata_fast/contents/$Language.traineddata?ref=main"
    $Metadata = Invoke-RestMethod -Uri $MetadataUrl
    if ((Test-Path -LiteralPath $Target) -and (Get-Item -LiteralPath $Target).Length -eq $Metadata.size) {
        continue
    }
    $BlobUrl = "https://api.github.com/repos/tesseract-ocr/tessdata_fast/git/blobs/$($Metadata.sha)"
    Invoke-WebRequest -Uri $BlobUrl -Headers @{ Accept = 'application/vnd.github.raw' } -OutFile $Target
    if ((Get-Item -LiteralPath $Target).Length -ne $Metadata.size) {
        throw "Downloaded language data failed the size check: $Language"
    }
}

$Languages = & $TesseractExecutable --list-langs --tessdata-dir $DataDirectory
if (($Languages -notcontains 'eng') -or ($Languages -notcontains 'chi_sim')) {
    throw 'Tesseract did not report both eng and chi_sim language data.'
}

Write-Host "Tesseract OCR ready: $TesseractExecutable"
Write-Host "Language data ready: $DataDirectory"
