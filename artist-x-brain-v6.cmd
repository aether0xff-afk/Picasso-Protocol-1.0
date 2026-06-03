@echo off
setlocal
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
"%CONDA%" run --no-capture-output -n picasso-gpu python "%~dp0artist_x_brain_v5.py" %* --model "%~dp0models\private_v6_ultradrop\artist_x_private_brain_fragment.pt"
exit /b %ERRORLEVEL%
