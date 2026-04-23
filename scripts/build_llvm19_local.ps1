# Builds a stock LLVM 19.1.x distribution locally on Windows using MSVC.
#
# Mirrors .github/workflows/build_llvm19_windows.yml but runs on the local
# machine so you can leverage fast hardware (e.g. Ryzen 9950X) and skip the
# 6-hour GitHub Actions timeout.
#
# Requirements (auto-detected where possible):
#   * Visual Studio 2026 with "Desktop development with C++" workload
#     (locates MSVC via vswhere.exe)
#   * CMake 3.20+ (bundled with VS works; or install separately)
#   * Ninja on PATH (falls back to CMake-discovered ninja in VS)
#   * git on PATH
#
# Typical usage (from repo root, in any shell):
#   pwsh -File scripts/build_llvm19_local.ps1
#
# Options:
#   -LlvmTag       LLVM git tag to check out (default llvmorg-19.1.7)
#   -SourceDir     Where to clone llvm-project (default .\build\llvm19-src)
#   -BuildDir      CMake build tree     (default .\build\llvm19-build)
#   -InstallDir    Install prefix       (default .\dist\taichi-llvm-19)
#   -ZipPath       Output zip           (default .\dist\taichi-llvm-19-msvc2026.zip)
#   -Jobs          Parallel compile jobs (default = logical CPU count)
#   -SkipZip       Skip packaging the zip (install-only)
#   -SkipClone     Reuse existing SourceDir (no fetch)
#   -Clean         Remove BuildDir & InstallDir before starting
#
# Output:
#   - InstallDir contains headers + libs + lib/cmake/llvm
#   - Setting LLVM_DIR=<InstallDir>\lib\cmake\llvm lets
#     `python build.py` use it without downloading anything.

[CmdletBinding()]
param(
    [string]$LlvmTag    = "llvmorg-19.1.7",
    [string]$SourceDir  = ".llvm19/src",
    [string]$BuildDir   = ".llvm19/build",
    [string]$InstallDir = "dist/taichi-llvm-19",
    [string]$ZipPath    = "dist/taichi-llvm-19-msvc2026.zip",
    [int]   $Jobs       = 0,
    [switch]$SkipZip,
    [switch]$SkipClone,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Windows PowerShell 5.1 (default powershell.exe on Windows) treats any
# stderr output from native commands as an error when $ErrorActionPreference
# is set to Stop. That means git's informative "Cloning into..." message,
# CMake's warnings, and compiler diagnostics all abort the script. The
# helpers below temporarily relax that behaviour and rely on explicit
# $LASTEXITCODE checks instead.

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArgs
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @RemainingArgs
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe exited with code $LASTEXITCODE"
    }
}

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# Resolve to absolute paths relative to current location
$SourceDir  = (Join-Path (Get-Location) $SourceDir)
$BuildDir   = (Join-Path (Get-Location) $BuildDir)
$InstallDir = (Join-Path (Get-Location) $InstallDir)
$ZipPath    = (Join-Path (Get-Location) $ZipPath)

if ($Jobs -le 0) {
    $Jobs = [Environment]::ProcessorCount
}

Write-Step "Configuration"
Write-Host "  LlvmTag    : $LlvmTag"
Write-Host "  SourceDir  : $SourceDir"
Write-Host "  BuildDir   : $BuildDir"
Write-Host "  InstallDir : $InstallDir"
Write-Host "  ZipPath    : $ZipPath"
Write-Host "  Jobs       : $Jobs"

# ---------------------------------------------------------------------------
# 0. Fast path: if the install tree already looks complete, skip the entire
#    build. LLVM 19 is a ~30 minute compile; re-running it every time a
#    caller invokes this script is expensive and has caused
#    "ninja: error: failed recompaction: Permission denied" incidents
#    when two concurrent invocations fought over .ninja_log.
# ---------------------------------------------------------------------------
$llvmCMakeProbe = Join-Path $InstallDir "lib\cmake\llvm\LLVMConfig.cmake"
$llvmCoreProbe  = Join-Path $InstallDir "lib\LLVMCore.lib"
if ((Test-Path $llvmCMakeProbe) -and (Test-Path $llvmCoreProbe) -and -not $Clean) {
    Write-Step "Existing install detected — skipping build"
    Write-Host "  Found     : $llvmCMakeProbe"
    Write-Host "  Found     : $llvmCoreProbe"
    Write-Host ""
    Write-Host "Set LLVM_DIR and build Taichi:"
    Write-Host "  `$env:LLVM_DIR = '$(Join-Path $InstallDir 'lib\cmake\llvm')'"
    Write-Host "  python build.py --python 3.10"
    Write-Host ""
    Write-Host "Pass -Clean to force a full rebuild."
    exit 0
}

# ---------------------------------------------------------------------------
# 1. Locate MSVC via vswhere and enter the x64 dev environment.
# ---------------------------------------------------------------------------
function Enter-MsvcDevEnv {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        $vswhere = "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    }
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio 2026 with the 'Desktop development with C++' workload."
    }

    $vsRoot = & $vswhere -latest -prerelease `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsRoot) {
        throw "No VS installation with MSVC x64 tools found."
    }
    Write-Host "  VS root    : $vsRoot"

    $vsDevCmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        throw "VsDevCmd.bat not found at $vsDevCmd"
    }

    # Dump env vars set by VsDevCmd into the current PowerShell session.
    $envDump = cmd /c "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 -no_logo && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
        }
    }
}

Write-Step "Locate MSVC toolchain"
Enter-MsvcDevEnv

# Verify cl & cmake reachable
$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue)
if (-not $cl) { throw "cl.exe not on PATH after VsDevCmd — toolchain missing." }
Write-Host "  cl.exe     : $($cl.Path)"
$cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
if (-not $cmake) { throw "cmake.exe not on PATH. Install CMake 3.20+ or enable the VS CMake component." }
Write-Host "  cmake.exe  : $($cmake.Path)"

# Ninja is required (Ninja generator is what the workflow uses)
$ninja = (Get-Command ninja.exe -ErrorAction SilentlyContinue)
if (-not $ninja) {
    # VS ships ninja under the CMake component; VsDevCmd usually puts it on PATH.
    throw "ninja.exe not on PATH. Install via winget (`"winget install Ninja-build.Ninja`") or enable the VS CMake component."
}
Write-Host "  ninja.exe  : $($ninja.Path)"

# ---------------------------------------------------------------------------
# 2. Clean (optional)
# ---------------------------------------------------------------------------
if ($Clean) {
    Write-Step "Clean previous build/install dirs"
    foreach ($d in @($BuildDir, $InstallDir)) {
        if (Test-Path $d) {
            Write-Host "  removing $d"
            [System.IO.Directory]::Delete($d, $true)
        }
    }
}

# ---------------------------------------------------------------------------
# 3. Clone LLVM (shallow) unless -SkipClone
# ---------------------------------------------------------------------------
if (-not $SkipClone) {
    $llvmSub = Join-Path $SourceDir "llvm"
    if (-not (Test-Path $SourceDir)) {
        Write-Step "Clone llvm-project @ $LlvmTag"
        Invoke-Native git clone --depth 1 --branch $LlvmTag https://github.com/llvm/llvm-project.git $SourceDir
    } elseif (-not (Test-Path $llvmSub)) {
        # Source dir exists but no llvm/ subtree — likely an aborted clone
        # (just the .git folder). Wipe and re-clone rather than failing later
        # with a confusing "source directory does not exist" CMake error.
        Write-Host "  $SourceDir exists but has no llvm/ subtree — re-cloning."
        Remove-Item -Recurse -Force $SourceDir
        Invoke-Native git clone --depth 1 --branch $LlvmTag https://github.com/llvm/llvm-project.git $SourceDir
    } else {
        Write-Host "  $SourceDir already exists — reusing (pass -SkipClone to silence this, or delete the dir)."
    }
} else {
    if (-not (Test-Path $SourceDir)) {
        throw "-SkipClone specified but $SourceDir does not exist."
    }
}

# ---------------------------------------------------------------------------
# 4. Configure (CMake flags mirror .github/workflows/build_llvm19_windows.yml)
# ---------------------------------------------------------------------------
Write-Step "Configure LLVM via CMake"
$cmakeArgs = @(
    "-S", (Join-Path $SourceDir "llvm"),
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$InstallDir",
    "-DLLVM_ENABLE_PROJECTS=",
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU",
    "-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=DirectX",
    "-DLLVM_ENABLE_RTTI=ON",
    "-DLLVM_ENABLE_ASSERTIONS=OFF",
    "-DLLVM_ENABLE_TERMINFO=OFF",
    "-DLLVM_ENABLE_ZSTD=OFF",
    "-DLLVM_ENABLE_ZLIB=OFF",
    "-DLLVM_ENABLE_LIBXML2=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
    "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
    "-DLLVM_BUILD_LLVM_C_DYLIB=OFF",
    "-DLLVM_BUILD_TOOLS=OFF",
    "-DLLVM_BUILD_UTILS=OFF",
    "-DLLVM_HOST_TRIPLE=x86_64-pc-windows-msvc"
)
Invoke-Native cmake @cmakeArgs

# ---------------------------------------------------------------------------
# 5. Build & install
# ---------------------------------------------------------------------------
Write-Step "Build & install (parallel=$Jobs)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Native cmake --build $BuildDir --target install --config Release --parallel $Jobs
$sw.Stop()
Write-Host ("  Build time : {0:N1} minutes" -f $sw.Elapsed.TotalMinutes)

# ---------------------------------------------------------------------------
# 6. Package zip (optional)
# ---------------------------------------------------------------------------
if (-not $SkipZip) {
    Write-Step "Package zip"
    $zipDir = Split-Path $ZipPath -Parent
    if (-not (Test-Path $zipDir)) { New-Item -ItemType Directory -Path $zipDir | Out-Null }
    if (Test-Path $ZipPath) { [System.IO.File]::Delete($ZipPath) }
    Compress-Archive -Path (Join-Path $InstallDir '*') -DestinationPath $ZipPath
    $hash = Get-FileHash $ZipPath -Algorithm SHA256
    $hash.Hash | Out-File -Encoding ASCII "$ZipPath.sha256.txt"
    Write-Host "  zip       : $ZipPath"
    Write-Host "  sha256    : $($hash.Hash)"
}

# ---------------------------------------------------------------------------
# 7. Print next-step hints
# ---------------------------------------------------------------------------
$llvmCMakeDir = Join-Path $InstallDir "lib\cmake\llvm"
Write-Step "Done."
Write-Host "LLVM 19 installed at: $InstallDir"
Write-Host ""
Write-Host "Point Taichi at the local build by exporting LLVM_DIR before calling build.py:"
Write-Host ""
Write-Host "  `$env:LLVM_DIR = '$llvmCMakeDir'"
Write-Host "  python build.py --python 3.10"
Write-Host ""
if (-not $SkipZip) {
    Write-Host "Or host the zip locally and set LLVM19_WIN_URL to a file:/// URL if you prefer the download path."
}
