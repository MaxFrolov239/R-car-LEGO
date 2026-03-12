$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

$srcRoot     = "C:\src"
$opencvVer   = "4.12.0"
$opencvZip   = Join-Path $srcRoot "opencv-$opencvVer.zip"
$opencvSrc   = Join-Path $srcRoot "opencv-$opencvVer"

$contribZip  = Join-Path $srcRoot "opencv_contrib-$opencvVer.zip"
$contribSrc  = Join-Path $srcRoot "opencv_contrib-$opencvVer"
$cudevPath   = Join-Path $contribSrc "modules\cudev"

$buildDir    = Join-Path $opencvSrc "build_cuda"
$installDir  = "C:\OpenCV-cuda\install"

$cudnnRoot    = "C:\tools\cudnn"
$cudnnInclude = Join-Path $cudnnRoot "include"
$cudnnLib     = Join-Path $cudnnRoot "lib\x64\cudnn.lib"

$cudaArchBin = "8.6"

Ensure-Dir $srcRoot
Ensure-Dir $installDir

if (-not (Test-Path $cudnnInclude)) {
    throw "Не найдена папка cuDNN include: $cudnnInclude"
}
if (-not (Test-Path $cudnnLib)) {
    throw "Не найден файл cuDNN library: $cudnnLib"
}

if (-not (Test-Path $opencvSrc)) {
    Write-Host "Скачиваю OpenCV $opencvVer..."
    Invoke-WebRequest -Uri "https://github.com/opencv/opencv/archive/refs/tags/$opencvVer.zip" -OutFile $opencvZip
    Expand-Archive -Path $opencvZip -DestinationPath $srcRoot -Force
}

if (-not (Test-Path $contribSrc)) {
    Write-Host "Скачиваю opencv_contrib $opencvVer..."
    Invoke-WebRequest -Uri "https://github.com/opencv/opencv_contrib/archive/refs/tags/$opencvVer.zip" -OutFile $contribZip
    Expand-Archive -Path $contribZip -DestinationPath $srcRoot -Force
}

if (-not (Test-Path $cudevPath)) {
    throw "Не найден модуль cudev: $cudevPath"
}

if (Test-Path $buildDir) {
    Remove-Item $buildDir -Recurse -Force
}
Ensure-Dir $buildDir

$cmakeArgs = @(
    "-S", $opencvSrc,
    "-B", $buildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DBUILD_opencv_world=ON",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_opencv_python3=OFF",
    "-DWITH_CUDA=ON",
    "-DWITH_CUDNN=ON",
    "-DOPENCV_DNN_CUDA=ON",
    "-DOPENCV_EXTRA_MODULES_PATH=$cudevPath",
    "-DCUDNN_INCLUDE_DIR=$cudnnInclude",
    "-DCUDNN_LIBRARY=$cudnnLib",
    "-DCUDA_ARCH_BIN=$cudaArchBin",
    "-DWITH_MSMF=ON",
    "-DWITH_DSHOW=ON"
)

Write-Host "Конфигурирую OpenCV + cudev..."
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure завершился с ошибкой: $LASTEXITCODE"
}

Write-Host "Собираю и устанавливаю OpenCV..."
& cmake --build $buildDir --config Release --target INSTALL --parallel
if ($LASTEXITCODE -ne 0) {
    throw "CMake build/install завершился с ошибкой: $LASTEXITCODE"
}

$cudnnBin = Join-Path $cudnnRoot "bin"
$opencvBin = Join-Path $installDir "x64\vc17\bin"
if (Test-Path (Join-Path $cudnnBin "cudnn64_8.dll")) {
    Copy-Item (Join-Path $cudnnBin "cudnn*.dll") $opencvBin -Force
    Write-Host "Скопированы cuDNN DLL в: $opencvBin"
} else {
    Write-Warning "Не найден cudnn64_8.dll в: $cudnnBin"
    Write-Warning "Скопируй cudnn*.dll вручную в: $opencvBin"
}

Write-Host "Готово: $installDir"
