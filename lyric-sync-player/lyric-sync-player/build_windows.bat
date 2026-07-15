@echo off
REM Build a standalone Windows .exe for the Lyric Sync Player.
REM Run this ON A WINDOWS MACHINE (PyInstaller can't cross-compile from Linux/Mac).

echo Installing build dependencies...
pip install --upgrade pyinstaller pygame

echo.
echo Building LyricSyncPlayer.exe ...
pyinstaller --onefile --console --name LyricSyncPlayer lyric_sync.py

echo.
echo Copying sample lyric file next to the exe...
if not exist "dist\media" mkdir "dist\media"
copy /Y "media\sample.lrc" "dist\media\sample.lrc" >nul

echo.
echo Done. Everything the recipient needs is in the "dist" folder:
echo   dist\LyricSyncPlayer.exe
echo   dist\media\sample.lrc        (demo - they can delete/replace it)
echo.
echo When they run the exe, it looks for audio + .lrc files inside a
echo "media" folder sitting right next to it (it auto-creates the folder
echo the first time it's run if missing).
pause
