@echo off
setlocal
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
"%CONDA%" run --no-capture-output -n picasso-gpu python "%~dp0artist_x_private.py" decode %* --model "%~dp0models\private_v4\artist_x_private_decoder.pt"
exit /b %ERRORLEVEL%
