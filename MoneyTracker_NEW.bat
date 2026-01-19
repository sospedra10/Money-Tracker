@echo off
setlocal

cd /d "%~dp0"

echo === Money Tracker (NEW) ===
echo.

if not exist "venv\\Scripts\\python.exe" (
  echo ERROR: No se encuentra `venv\\Scripts\\python.exe`.
  echo Crea el entorno virtual primero (o ajusta el script).
  pause
  exit /b 1
)

echo [1/3] Backend deps...
"venv\\Scripts\\python.exe" -m pip install -r "backend\\requirements.txt"
if errorlevel 1 goto :error

echo [2/3] Frontend deps + build...
pushd "frontend" >nul
if not exist "node_modules" (
  npm install
  if errorlevel 1 goto :errorpop
)
if not exist "dist\\index.html" (
  npm run build
  if errorlevel 1 goto :errorpop
)
popd >nul

echo [3/3] Start server...
pushd "backend" >nul
start "MoneyTracker Backend" cmd /k ..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
popd >nul

timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"

echo Listo: http://127.0.0.1:8000
exit /b 0

:errorpop
popd >nul
:error
echo.
echo ERROR: Algo ha fallado. Mira el output arriba.
pause
exit /b 1
