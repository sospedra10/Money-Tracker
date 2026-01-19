@echo off
setlocal

cd /d "%~dp0"

echo === Money Tracker (DEV) ===
echo Backend: http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.

if not exist "venv\\Scripts\\python.exe" (
  echo ERROR: No se encuentra `venv\\Scripts\\python.exe`.
  pause
  exit /b 1
)

"venv\\Scripts\\python.exe" -m pip install -r "backend\\requirements.txt"
if errorlevel 1 goto :error

pushd "backend" >nul
start "MoneyTracker Backend" cmd /k ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
popd >nul

pushd "frontend" >nul
if not exist "node_modules" (
  npm install
  if errorlevel 1 goto :errorpop
)
start "MoneyTracker Frontend" cmd /k "npm run dev"
popd >nul

timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5173"

exit /b 0

:errorpop
popd >nul
:error
echo.
echo ERROR: Algo ha fallado. Mira el output arriba.
pause
exit /b 1
