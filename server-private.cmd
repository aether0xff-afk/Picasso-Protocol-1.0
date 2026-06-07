@echo off
setlocal
set "PICASSO_ENABLE_PRIVATE_DECODE=1"
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0toWebPage\server.py"
exit /b %ERRORLEVEL%
