@echo off
setlocal
chcp 65001 >nul
cd /d %~dp0

python -m PyInstaller --noconfirm --clean FraudShield.spec
if errorlevel 1 (
  echo.
  echo Build failed. Install build dependencies first:
  echo     python -m pip install -r requirements-build.txt
  pause
  exit /b 1
)

echo.
echo Build completed. Output folder:
 echo    dist\FraudShield
pause
