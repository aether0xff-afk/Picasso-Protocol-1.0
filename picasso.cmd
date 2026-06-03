@echo off
setlocal
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA%" (
  echo Conda executable not found: %CONDA% 1>&2
  exit /b 1
)
"%CONDA%" run --no-capture-output -n picasso-gpu python "%~dp0picasso.py" %*
exit /b %ERRORLEVEL%
