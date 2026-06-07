@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0tools\cli\artist_x_private.py" decode %* --model "%~dp0models\private_v4\artist_x_private_decoder.pt"
exit /b %ERRORLEVEL%
