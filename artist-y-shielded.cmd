@echo off
setlocal
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
"%CONDA%" run --no-capture-output -n picasso-gpu python "%~dp0artist_y_public.py" encode %* --model "%~dp0models\public_v4\artist_y_public_encoder.pt" --shield-dropout 0.65 --shield-noise 1.0
exit /b %ERRORLEVEL%
