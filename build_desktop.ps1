$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

Write-Host "=== DDQ-RAG Workbench Build ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. Install dependencies ──────────────────────────────────────────────
Write-Host "[1/4] Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install -r "$ScriptDir\requirements.txt" --quiet
Write-Host "  Dependencies OK" -ForegroundColor Green

# ── 2. Ensure data directory ─────────────────────────────────────────────
$dataDir = Join-Path $ScriptDir "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

# ── 3. PyInstaller build ─────────────────────────────────────────────────
Write-Host "[2/4] Running PyInstaller..." -ForegroundColor Yellow

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name "DDQ-RAG-Workbench" `
  --icon=NONE `
  --add-data "$ScriptDir\RAGmaterial;RAGmaterial" `
  --add-data "$ScriptDir\data;data" `
  --hidden-import "docx" `
  --hidden-import "PyPDF2" `
  --hidden-import "openpyxl" `
  --hidden-import "pydantic" `
  --hidden-import "fastapi" `
  --hidden-import "uvicorn" `
  --hidden-import "sentence_transformers" `
  --hidden-import "transformers" `
  --hidden-import "torch" `
  --hidden-import "sklearn" `
  --hidden-import "numpy" `
  --hidden-import "scipy" `
  --hidden-import "PIL" `
  --hidden-import "tokenizers" `
  --hidden-import "huggingface_hub" `
  --hidden-import "pyarrow" `
  --hidden-import "app.config" `
  --hidden-import "app.embeddings" `
  --hidden-import "app.ingest" `
  --hidden-import "app.retrieve" `
  --hidden-import "app.generate" `
  --hidden-import "app.storage" `
  --hidden-import "app.confidence" `
  --hidden-import "app.library" `
  --hidden-import "app.excel_processor" `
  --hidden-import "app.prompts" `
  --hidden-import "app.schemas" `
  --hidden-import "app.text_cleaner" `
  --hidden-import "app.metadata_filters" `
  --hidden-import "app.ranking" `
  --hidden-import "app.reranker" `
  --hidden-import "app.evaluation" `
  --hidden-import "pandas" `
  --exclude-module "jupyter" `
  --exclude-module "notebook" `
  --exclude-module "IPython" `
  --exclude-module "matplotlib" `
  desktop/client.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "  PyInstaller failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  PyInstaller OK" -ForegroundColor Green

# ── 4. Copy runtime files ────────────────────────────────────────────────
Write-Host "[3/4] Copying runtime files..." -ForegroundColor Yellow

$distDir = Join-Path $ScriptDir "dist\DDQ-RAG-Workbench"

# Copy .env if it exists (API keys, model path), otherwise .env.example
# Place next to exe (for cwd-based discovery) AND inside _internal (for MEIPASS-based discovery)
$envSource = Join-Path $ScriptDir ".env"
$envDestExe = Join-Path $distDir ".env"
$envDestInternal = Join-Path $distDir "_internal\.env"
if (Test-Path $envSource) {
    Copy-Item -LiteralPath $envSource -Destination $envDestExe -Force
    Copy-Item -LiteralPath $envSource -Destination $envDestInternal -Force
    Write-Host "  .env copied (project root + _internal)" -ForegroundColor Gray
} elseif (Test-Path (Join-Path $ScriptDir ".env.example")) {
    Copy-Item -LiteralPath (Join-Path $ScriptDir ".env.example") -Destination $envDestExe -Force
    Copy-Item -LiteralPath (Join-Path $ScriptDir ".env.example") -Destination $envDestInternal -Force
    Write-Host "  .env.example copied (edit with your keys)" -ForegroundColor DarkYellow
}

# Copy RAGmaterial (if not bundled already by --add-data)
$ragTarget = Join-Path $distDir "RAGmaterial"
if (-not (Test-Path $ragTarget)) {
    Copy-Item -LiteralPath (Join-Path $ScriptDir "RAGmaterial") -Destination $ragTarget -Recurse -Force
    Write-Host "  RAGmaterial/ copied" -ForegroundColor Gray
}

# Copy data/ddqrag.sqlite3 if it exists
$dbSource = Join-Path $ScriptDir "data\ddqrag.sqlite3"
$dataTarget = Join-Path $distDir "data"
if (-not (Test-Path $dataTarget)) {
    New-Item -ItemType Directory -Path $dataTarget | Out-Null
}
if (Test-Path $dbSource) {
    Copy-Item -LiteralPath $dbSource -Destination (Join-Path $dataTarget "ddqrag.sqlite3") -Force
    Write-Host "  Database copied" -ForegroundColor Gray
}

Write-Host "  Runtime files OK" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Output:  " -NoNewline
Write-Host (Join-Path $distDir "DDQ-RAG-Workbench.exe") -ForegroundColor Cyan
Write-Host ""
Write-Host "  Before running:" -ForegroundColor Yellow
Write-Host "  1. Edit .env in the dist folder to set:"
Write-Host "     - BGE_MODEL_PATH  (absolute path to BGE-M3 model)"
Write-Host "     - GEMINI_API_KEY  (if using Gemini generation)"
Write-Host "  2. Place documents in the RAGmaterial/ folder"
Write-Host "  3. Launch DDQ-RAG-Workbench.exe"
Write-Host ""
