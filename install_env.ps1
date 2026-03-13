# ============================================================================
# install_env.ps1 — Pokemon Stencil Art Factory — Windows fresh environment
# ============================================================================
#
# Usage (run from the repo root in PowerShell):
#   .\install_env.ps1           # CPU-only PyTorch
#   .\install_env.ps1 -CUDA     # CUDA 11.8 PyTorch (recommended for GPU)
#   .\install_env.ps1 -CUDA121  # CUDA 12.1 PyTorch
#
# Requirements:
#   - Python 3.11 on PATH
#   - Internet access (downloads from PyPI and pytorch.org)
# ============================================================================

param(
    [switch]$CUDA,
    [switch]$CUDA121
)

$ErrorActionPreference = "Stop"
$VENV_DIR = ".venv"
$PYTHON = "python"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Pokemon Stencil Art Factory — Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Verify Python 3.11 ───────────────────────────────────────────────
Write-Host "[1/7] Checking Python version..." -ForegroundColor Yellow
$pyver = & $PYTHON --version 2>&1
Write-Host "      Found: $pyver"
if ($pyver -notmatch "Python 3\.11") {
    Write-Host "ERROR: Python 3.11 is required. Found: $pyver" -ForegroundColor Red
    Write-Host "       Download from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# ── Step 2: Create virtual environment ───────────────────────────────────────
Write-Host "[2/7] Creating virtual environment in $VENV_DIR ..." -ForegroundColor Yellow
if (Test-Path $VENV_DIR) {
    Write-Host "      Removing existing venv..."
    Remove-Item -Recurse -Force $VENV_DIR
}
& $PYTHON -m venv $VENV_DIR
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: venv creation failed." -ForegroundColor Red; exit 1 }

$PIP = "$VENV_DIR\Scripts\pip.exe"
$PYPROG = "$VENV_DIR\Scripts\python.exe"

# ── Step 3: Upgrade pip / setuptools / wheel ─────────────────────────────────
Write-Host "[3/7] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
& $PYPROG -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip upgrade failed." -ForegroundColor Red; exit 1 }

# ── Step 4: Install PyTorch (CPU or CUDA) ────────────────────────────────────
Write-Host "[4/7] Installing PyTorch..." -ForegroundColor Yellow
if ($CUDA121) {
    Write-Host "      Installing CUDA 12.1 build..."
    & $PIP install torch==2.1.2 torchvision==0.16.2 `
        --index-url https://download.pytorch.org/whl/cu121
} elseif ($CUDA) {
    Write-Host "      Installing CUDA 11.8 build..."
    & $PIP install torch==2.1.2 torchvision==0.16.2 `
        --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "      Installing CPU-only build (use -CUDA for GPU support)..."
    & $PIP install torch==2.1.2 torchvision==0.16.2 `
        --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: PyTorch installation failed." -ForegroundColor Red; exit 1 }

# ── Step 5: Install remaining requirements ───────────────────────────────────
Write-Host "[5/7] Installing project requirements from requirements.txt..." -ForegroundColor Yellow
& $PIP install -r requirements.txt --no-deps-update
if ($LASTEXITCODE -ne 0) {
    Write-Host "      Retrying without --no-deps-update..." -ForegroundColor Yellow
    & $PIP install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: requirements installation failed." -ForegroundColor Red; exit 1
    }
}

# ── Step 6: Install the project itself (editable) ────────────────────────────
Write-Host "[6/7] Installing pokemon-stencil-factory in editable mode..." -ForegroundColor Yellow
& $PIP install -e . --no-deps
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: project install failed." -ForegroundColor Red; exit 1 }

# ── Step 7: Verify key versions ──────────────────────────────────────────────
Write-Host "[7/7] Version verification..." -ForegroundColor Yellow
& $PYPROG -c @"
import sys
print(f'Python: {sys.version}')
import torch; print(f'torch: {torch.__version__}')
import diffusers; print(f'diffusers: {diffusers.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import accelerate; print(f'accelerate: {accelerate.__version__}')
import peft; print(f'peft: {peft.__version__}')
import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import safetensors; print(f'safetensors: {safetensors.__version__}')
import cv2; print(f'opencv: {cv2.__version__}')
import PIL; print(f'Pillow: {PIL.__version__}')
print('All core imports OK.')
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Installation complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the venv:     .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Verify environment:    python verify_env.py"
Write-Host "  3. Run tests:             python run_all_tests.py"
Write-Host "  4. Download models:       python -c `"from pokemon_stencil.models.model_manager import ModelManager; ModelManager().ensure_models()`""
Write-Host "  5. Place SDXL LoRA:       Copy an SDXL .safetensors LoRA to models\lora\"
Write-Host "  6. Launch dashboard:      streamlit run pokemon_stencil\dashboard\app.py"
Write-Host ""
Write-Host "GPU notes:" -ForegroundColor Yellow
Write-Host "  - For CUDA: re-run with .\install_env.ps1 -CUDA (CUDA 11.8)"
Write-Host "  - For CUDA 12.1: re-run with .\install_env.ps1 -CUDA121"
Write-Host "  - Set torch_dtype='float16' in config for GPU acceleration"
Write-Host ""
