@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0tools\cli\artist_y_public.py" encode %* --model "%~dp0models\public_v4\artist_y_public_encoder.pt" --shield-dropout 0.65 --shield-noise 1.0
exit /b %ERRORLEVEL%
