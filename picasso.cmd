@echo off
setlocal
if defined CONDA_PREFIX (
  set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
  set "PYTHON=python"
)
if not exist "%PYTHON%" if not "%PYTHON%"=="python" (
  echo Python executable not found: %PYTHON% 1>&2
  exit /b 1
)
"%PYTHON%" "%~dp0tools\cli\picasso.py" %*
exit /b %ERRORLEVEL%
