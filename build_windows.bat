@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv-win" (
  python -m venv .venv-win
)

call .venv-win\Scripts\python -m pip install --upgrade pip
call .venv-win\Scripts\pip install -r requirements-gui.txt

call .venv-win\Scripts\pyinstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name BottleDefectDetector ^
  --collect-all PySide6 ^
  run_gui.py

if exist "dist\BottleDefectDetector" (
  if exist "assets\reference_bottles" (
    mkdir "dist\BottleDefectDetector\assets\reference_bottles" >nul 2>nul
    copy /Y "assets\reference_bottles\*.jpg" "dist\BottleDefectDetector\assets\reference_bottles\" >nul
    echo Copied AI reference bottle images into dist\BottleDefectDetector\assets\reference_bottles
  )
  copy /Y ".env.example" "dist\BottleDefectDetector\.env.example" >nul
  if exist ".env" (
    copy /Y ".env" "dist\BottleDefectDetector\.env" >nul
    echo Copied current .env into dist\BottleDefectDetector\.env
  ) else (
    copy /Y ".env.example" "dist\BottleDefectDetector\.env" >nul
    echo No .env found, copied .env.example as dist\BottleDefectDetector\.env
  )
  echo.
  echo Build complete:
  echo dist\BottleDefectDetector\BottleDefectDetector.exe
  echo.
  echo Send the whole dist\BottleDefectDetector folder to the operator.
  echo The folder includes .env, outputs, and result folders will be created beside the EXE.
)

endlocal
