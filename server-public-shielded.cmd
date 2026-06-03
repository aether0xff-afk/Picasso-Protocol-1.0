@echo off
setlocal
set "PICASSO_ENABLE_PRIVATE_DECODE="
set "PICASSO_PUBLIC_MODEL=%~dp0models\public_v4\artist_y_public_encoder.pt"
set "PICASSO_PUBLIC_DROPOUT=0.65"
set "PICASSO_PUBLIC_NOISE=1.0"
set "PYTHON=%USERPROFILE%\miniconda3\envs\picasso-gpu\python.exe"
"%PYTHON%" "%~dp0toWebPage\server.py"
exit /b %ERRORLEVEL%
