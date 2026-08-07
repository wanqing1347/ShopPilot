$ErrorActionPreference = "Stop"

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BasePython = "D:\Software\Python\python.exe"
$VenvDir = Join-Path $ProjectRoot ".venv312"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$DatasetRoot = Join-Path $ProjectRoot "data\merged_catalog"

if (-not (Test-Path $BasePython)) {
    throw "Python 3.12 executable not found: $BasePython"
}

Push-Location $ProjectRoot
try {
    Write-Host "[1/10] Confirming requested Python interpreter..."
    & $BasePython -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable); print(sys.version)"
    Assert-LastExit "Python 3.12 check"

    if (-not (Test-Path $Python)) {
        Write-Host "[2/10] Creating Python 3.12 virtual environment..."
        & $BasePython -m venv $VenvDir
        Assert-LastExit "Virtual environment creation"
    }
    else {
        Write-Host "[2/10] Reusing Python 3.12 virtual environment..."
    }

    Write-Host "[3/10] Installing project and test dependencies..."
    & $Python -m pip install --upgrade pip
    Assert-LastExit "pip upgrade"
    & $Python -m pip install -e ".[dev,postgres,retrieval,embedding]"
    Assert-LastExit "Project dependency installation"

    Write-Host "[4/10] Validating schema-v2 synthetic dataset..."
    if (-not (Test-Path $DatasetRoot)) {
        throw "Dataset directory not found: $DatasetRoot"
    }
    & $Python -c "from app.recall.catalog import load_catalog; rows = load_catalog(); print(f'validated {len(rows)} ShopPilot catalog items')"
    Assert-LastExit "Dataset validation"

    Write-Host "[5/10] Compiling Python sources..."
    & $Python -m compileall -q app tests scripts
    Assert-LastExit "Python compileall"

    Write-Host "[6/10] Checking LangGraph construction with memory checkpoint..."
    $env:SHOPPILOT_CHECKPOINT_BACKEND = "memory"
    & $Python scripts/verify_runtime.py
    Assert-LastExit "LangGraph construction"

    Write-Host "[7/10] Running tests, including BGE/LTR, knowledge retrieval and SQLite restart recovery..."
    & $Python -m pytest -q
    Assert-LastExit "pytest"

    Write-Host "[8/10] Running locked BGE + LTR test-set evaluation..."
    $env:SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER = "sentence_transformers"
    $env:SHOPPILOT_RETRIEVAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    $env:SHOPPILOT_RETRIEVAL_BM25_WEIGHT = "1.5"
    $env:SHOPPILOT_RETRIEVAL_VECTOR_WEIGHT = "0.25"
    $env:SHOPPILOT_RETRIEVAL_RERANK_WEIGHT = "0.25"
    $env:SHOPPILOT_RETRIEVAL_RERANKER = "auto"
    & $Python scripts/evaluate_retrieval.py --split test --k 5 10 20
    Assert-LastExit "Retrieval evaluation"

    Write-Host "[9/10] Running category knowledge retrieval evaluation..."
    & $Python scripts/evaluate_knowledge.py --k 1 3 5
    Assert-LastExit "Knowledge retrieval evaluation"

    Write-Host "[10/10] Building React frontend..."
    Push-Location frontend
    try {
        npm run build
        Assert-LastExit "Frontend build"
    }
    finally {
        Pop-Location
    }

    if ($env:RUN_LIVE_RAG_TEST -eq "1") {
        Write-Host "[live-rag] Running grounded CategoryInsight evaluation..."
        $env:SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED = "true"
        & $Python scripts/evaluate_grounded_rag.py
        Assert-LastExit "Grounded RAG evaluation"
    }
    else {
        Write-Host "Grounded RAG test skipped. Set RUN_LIVE_RAG_TEST=1 to enable it."
    }

    if ($env:RUN_LIVE_AGENT_TEST -eq "1") {
        Write-Host "[live-agent] Running one real AgentLoop smoke test..."
        $env:SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED = "true"
        Remove-Item Env:SHOPPILOT_CHECKPOINT_BACKEND -ErrorAction SilentlyContinue
        & $Python scripts/demo.py "Compare ceramic or glass coffee cups under CNY 300 across four platforms; exclude plastic and explain landed-price evidence."
        Assert-LastExit "Live AgentLoop smoke test"
    }
    else {
        Write-Host "Live Agent test skipped. Set RUN_LIVE_AGENT_TEST=1 to enable it."
    }

    Write-Host "Verification completed successfully with Python 3.12."
}
finally {
    Pop-Location
}
