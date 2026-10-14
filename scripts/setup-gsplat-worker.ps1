# Set up a Windows CUDA machine as a GaussQueue gsplat worker (trainer + pruning).
#
#   powershell -ExecutionPolicy Bypass -File setup-gsplat-worker.ps1 [-Root D:\gaussqueue]
#
# Installs, all under <Root> (nothing global, the system Python is untouched):
#   tools\uv       uv (pinned, checksum-verified) to fetch Python and packages
#   tools\pythons  Python 3.10 (gsplat's prebuilt Windows wheels are cp310 only)
#   gsplat\venv    torch 2.4.1 (CUDA 12.4) + gsplat 1.5.3 prebuilt (no compiler needed)
# Needs an NVIDIA driver recent enough for CUDA 12.4 (>= 550). The Mac copies the
# gs\*.py scripts to <Root>\gsplat\gs before every job, so there's nothing else to update.
param([string]$Root = "D:\gaussqueue")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$UvVersion = "0.12.21"
$UvSha256 = "5d223efa0bf00208c3853246af09420419dfbd352536aa6bb8163d6170e23890"   # uv-x86_64-pc-windows-msvc.zip
$Torch = "torch==2.4.1"
$TorchIndex = "https://download.pytorch.org/whl/cu124"
$Gsplat = "gsplat==1.5.3+pt24cu124"
$GsplatIndex = "https://docs.gsplat.studio/whl/gsplat/"

$uvDir = Join-Path $Root "tools\uv"
$uv = Join-Path $uvDir "uv.exe"
$env:UV_CACHE_DIR = Join-Path $Root "tools\uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\pythons"
$venv = Join-Path $Root "gsplat\venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $uv)) {
    New-Item -ItemType Directory -Force $uvDir | Out-Null
    $zip = Join-Path $uvDir "uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $UvSha256) { Remove-Item $zip; throw "uv checksum mismatch: $hash" }
    Expand-Archive $zip -DestinationPath $uvDir -Force
    Remove-Item $zip
}
& $uv --version

function Run([string[]]$a) {   # uv writes progress to stderr; don't treat that as failure
    $ErrorActionPreference = "Continue"
    & $uv @a 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE) { throw "uv $($a -join ' ') failed ($LASTEXITCODE)" }
}
Run @("python", "install", "3.10")
if (-not (Test-Path $py)) { Run @("venv", "--python", "3.10", $venv) }
Run @("pip", "install", "--python", $py, $Torch, "--index-url", $TorchIndex)
Run @("pip", "install", "--python", $py, $Gsplat, "--find-links", $GsplatIndex)
Run @("pip", "install", "--python", $py, "pillow", "packaging", "setuptools")
New-Item -ItemType Directory -Force (Join-Path $Root "gsplat\gs") | Out-Null

& $py -c "import torch, gsplat; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| gsplat', gsplat.__version__)"
if ($LASTEXITCODE) { throw "gsplat check failed" }

$r = $Root -replace '\\', '/'
Write-Host ""
Write-Host "Done. Add this to the worker's entry in the Mac's GaussQueue config.json:"
Write-Host "  `"gsplat`": {`"python`": `"$r/gsplat/venv/Scripts/python.exe`", `"scripts`": `"$r/gsplat/gs`"}"
