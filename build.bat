@echo off
echo ===========================================
echo Building AIDataTool for Windows...
echo ===========================================
pyinstaller AIDataTool.spec
echo ===========================================
echo Build finished! Executable: dist\AIDataTool.exe
echo ===========================================
pause
