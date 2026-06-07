@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0tools\cli\artist_x_brain_v5.py" %* --model "%~dp0models\private_v6_ultradrop\artist_x_private_brain_fragment.pt"
exit /b %ERRORLEVEL%
