# =============================================================================
# 量子RL调度系统 — Windows 一键环境初始化脚本 (PowerShell)
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -DevMode
# =============================================================================

param(
    [switch]$DevMode = $false,
    [switch]$NoVenv = $false
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Quantum RL Scheduler — Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Detect Python
# ---------------------------------------------------------------------------
Write-Host "[1/6] Checking Python..." -ForegroundColor White

$PythonExe = $null
$PyVersion = $null

# Try managed Python first
$managedPy = "C:\Users\$env:USERNAME\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if (Test-Path $managedPy) {
    $PythonExe = $managedPy
    $PyVersion = & $PythonExe --version 2>&1
    Write-Host "  [PASS] Found managed Python: $PyVersion" -ForegroundColor Green
}

# Fall back to system Python
if (-not $PythonExe) {
    foreach ($cmd in @("python3.12", "python3.11", "python3.10", "python3", "python")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            $versionStr = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            $major = [int]$versionStr.Split('.')[0]
            $minor = [int]$versionStr.Split('.')[1]
            if ($major -eq 3 -and $minor -ge 10) {
                $PythonExe = $cmd
                $PyVersion = & $PythonExe --version 2>&1
                Write-Host "  [PASS] Found system Python: $PyVersion" -ForegroundColor Green
                break
            }
        }
    }
}

if (-not $PythonExe) {
    Write-Host "  [FAIL] Python 3.10+ is required." -ForegroundColor Red
    Write-Host "  Download: https://www.python.org/downloads/"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
$VenvPath = ".venv"

if (-not $NoVenv) {
    Write-Host ""
    Write-Host "[2/6] Creating virtual environment..." -ForegroundColor White

    if (Test-Path $VenvPath) {
        Write-Host "  [INFO] Virtual environment already exists." -ForegroundColor Yellow
        $response = Read-Host "  Recreate? [y/N]"
        if ($response -eq "y" -or $response -eq "Y") {
            Remove-Item -Recurse -Force $VenvPath
            Write-Host "  Removing old virtual environment..."
        } else {
            Write-Host "  Using existing virtual environment."
        }
    }

    if (-not (Test-Path $VenvPath)) {
        & $PythonExe -m venv $VenvPath
        Write-Host "  [PASS] Virtual environment created." -ForegroundColor Green
    }

    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
} else {
    Write-Host ""
    Write-Host "[2/6] Skipped (--no-venv)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/6] Installing dependencies..." -ForegroundColor White

& $PythonExe -m pip install --upgrade pip --quiet

if ($DevMode) {
    Write-Host "  (dev mode: installing extra tools)"
    & $PythonExe -m pip install -r requirements.txt
    & $PythonExe -m pip install pre-commit pytest-watch debugpy bandit
} else {
    & $PythonExe -m pip install -r requirements.txt
}

Write-Host "  [PASS] Dependencies installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Configure .env
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/6] Setting up environment variables..." -ForegroundColor White

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "  [PASS] Created .env from .env.example." -ForegroundColor Green
    } else {
        @"
# Quantum RL Scheduler — Environment Configuration
TIANYAN_API_KEY=your_api_key_here
TIANYAN_API_SECRET=your_api_secret_here
TIANYAN_MOCK_MODE=true
LOG_LEVEL=INFO
"@ | Out-File -FilePath ".env" -Encoding utf8
        Write-Host "  [PASS] Created default .env." -ForegroundColor Green
    }
} else {
    Write-Host "  [INFO] .env already exists, keeping existing config." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Create directories
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/6] Creating project directories..." -ForegroundColor White

@("logs", "models", "data", "results") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}
Write-Host "  [PASS] Created: logs/ models/ data/ results/" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[6/6] Verifying installation..." -ForegroundColor White

$Modules = @(
    @("numpy", "numpy"),
    @("gymnasium", "gymnasium"),
    @("stable_baselines3", "stable-baselines3"),
    @("torch", "PyTorch"),
    @("qiskit", "Qiskit"),
    @("fastapi", "FastAPI"),
    @("sqlalchemy", "SQLAlchemy"),
    @("loguru", "Loguru"),
    @("pytest", "pytest"),
    @("black", "Black"),
    @("mypy", "mypy")
)

$PassCount = 0
$FailCount = 0

foreach ($mod in $Modules) {
    $importName = $mod[0]
    $displayName = $mod[1]
    $result = & $PythonExe -c "import $importName; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [PASS] $displayName" -ForegroundColor Green
        $PassCount++
    } else {
        Write-Host "  [FAIL] $displayName" -ForegroundColor Red
        $FailCount++
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Setup Complete! ($PassCount/$($PassCount+$FailCount) modules verified)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $NoVenv) {
    Write-Host "  Activate virtual environment:" -ForegroundColor White
    Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "  Quick Start:" -ForegroundColor White
Write-Host "    python -m pytest tests/ -v"
Write-Host "    python scripts/quick_train.py"
Write-Host "    python -m uvicorn src.visualization.app:app --host 0.0.0.0 --port 8000"
Write-Host "    python scripts/run_simulation.py"
Write-Host ""

if ($DevMode) {
    Write-Host "  Dev tools installed:" -ForegroundColor White
    Write-Host "    pre-commit install"
    Write-Host "    bandit -r src/"
    Write-Host ""
}

Write-Host "Happy coding!" -ForegroundColor Cyan
Write-Host ""
