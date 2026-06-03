@echo off
setlocal
set "PICASSO_ENABLE_PRIVATE_DECODE="
set "PYTHON=%USERPROFILE%\miniconda3\envs\picasso-gpu\python.exe"
"%PYTHON%" "%~dp0toWebPage\server.py"
exit /b %ERRORLEVEL%
