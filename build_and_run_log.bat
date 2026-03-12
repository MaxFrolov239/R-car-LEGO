@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".vscode\build_debug.cmd" (
  echo [ERROR] Build script not found: .vscode\build_debug.cmd
  exit /b 1
)

if not exist "logs" mkdir "logs"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "TS=%%I"
set "LOG_FILE=logs\run_%TS%.log"

echo [1/3] Building...
call ".vscode\build_debug.cmd"
if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

if not exist "usb_cam_app.exe" (
  echo [ERROR] Build finished but usb_cam_app.exe was not found.
  exit /b 1
)

echo [2/3] Starting app...
echo       Live log  : %LOG_FILE%
echo       Last log  : run_test.log
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; " ^
  "$log='%LOG_FILE%'; " ^
  "& '.\usb_cam_app.exe' 2>&1 | Tee-Object -FilePath $log"

set "APP_RC=%ERRORLEVEL%"

if exist "%LOG_FILE%" (
  copy /Y "%LOG_FILE%" "run_test.log" >nul
)

echo.
echo [3/3] Done. Exit code: %APP_RC%
echo       Saved log: %LOG_FILE%
echo       Copied to: run_test.log
exit /b %APP_RC%

