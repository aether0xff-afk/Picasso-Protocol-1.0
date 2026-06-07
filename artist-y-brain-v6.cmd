@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0tools\cli\artist_y_brain_v5.py" %* --model "%~dp0models\public_v6_ultradrop\artist_y_public_brain_fragment.pt"
exit /b %ERRORLEVEL%
