@echo off
setlocal
chcp 65001 >nul
cd /d %~dp0

set "ISCC_CMD="
for %%I in (ISCC.exe) do set "ISCC_CMD=%%~$PATH:I"

if not defined ISCC_CMD (
  if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_CMD=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if not defined ISCC_CMD (
  if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_CMD=C:\Program Files\Inno Setup 6\ISCC.exe"
)

if not defined ISCC_CMD (
  echo Inno Setup Compiler ISCC.exe was not found.
  echo Please install Inno Setup 6 or add ISCC.exe to PATH.
  pause
  exit /b 1
)

"%ISCC_CMD%" installer\FraudShield.iss
if errorlevel 1 (
  echo.
  echo Installer build failed.
  pause
  exit /b 1
)

echo.
echo Installer build completed. Output folder:
echo     dist_installer
pause
