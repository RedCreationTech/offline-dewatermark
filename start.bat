@echo off
rem Launcher for start.ps1 (Windows port of start.sh)
rem Usage: start.bat [setup|check-data|eval|test|test-release|freeze-baseline|control-inventory|control|report|gui] [args]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
exit /b %ERRORLEVEL%
