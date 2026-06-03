@echo off
setlocal
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
"%CONDA%" run --no-capture-output -n picasso-gpu python "%~dp0artist_y_brain_v5.py" %*
exit /b %ERRORLEVEL%
