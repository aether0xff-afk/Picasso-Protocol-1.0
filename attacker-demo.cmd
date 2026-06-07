@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" "%~dp0tools\attacks\attacker_demo.py" %*
exit /b %ERRORLEVEL%

