@echo off
setlocal

set "OPENCV_ROOT=C:\OpenCV-cuda\install"
set "OPENCV_INCLUDE=%OPENCV_ROOT%\include"
set "OPENCV_LIB_DIR=%OPENCV_ROOT%\x64\vc17\lib"
set "OPENCV_DEBUG_LIB=%OPENCV_LIB_DIR%\opencv_world4120d.lib"
set "OPENCV_RELEASE_LIB=%OPENCV_LIB_DIR%\opencv_world4120.lib"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

if not exist "%OPENCV_INCLUDE%\opencv2\opencv.hpp" (
  echo OpenCV headers not found: %OPENCV_INCLUDE%
  exit /b 1
)

if exist "%OPENCV_DEBUG_LIB%" (
  set "CRT_FLAG=/MDd"
  set "OPENCV_LIB_NAME=opencv_world4120d.lib"
) else if exist "%OPENCV_RELEASE_LIB%" (
  echo Debug OpenCV lib not found. Using Release lib.
  set "CRT_FLAG=/MD"
  set "OPENCV_LIB_NAME=opencv_world4120.lib"
) else (
  echo OpenCV libs not found in: %OPENCV_LIB_DIR%
  exit /b 1
)

cl /nologo /utf-8 /EHsc /std:c++17 /Zi %CRT_FLAG% /I "%OPENCV_INCLUDE%" roboshassi.cpp /Fe:usb_cam_app.exe /link /DEBUG /LIBPATH:"%OPENCV_LIB_DIR%" %OPENCV_LIB_NAME%
exit /b %errorlevel%
