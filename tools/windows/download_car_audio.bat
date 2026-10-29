@echo off
REM Engine Ear (cars side project) - step 2: download audio for the videos in car_urls.txt
REM Double-click. Audio goes to cars_download\raw_audio\<video_id>.<ext>. Re-running skips finished ones.
setlocal
cd /d "%~dp0"
call :find_ytdlp
if not defined YTDLP ( echo Could not find yt-dlp. & pause & exit /b 1 )
if not exist raw_audio mkdir raw_audio
"%YTDLP%" -f "ba[abr<=96]/ba" --no-playlist --ignore-errors --sleep-requests 1 ^
  --download-archive raw_audio\downloaded.txt ^
  -o "raw_audio\%%(id)s.%%(ext)s" -a car_urls.txt
echo.
echo Done. Tell Claude the download has finished.
pause
exit /b 0

:find_ytdlp
set "YTDLP="
for /f "delims=" %%p in ('where yt-dlp 2^>nul') do if not defined YTDLP set "YTDLP=%%p"
if not defined YTDLP if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\yt-dlp.exe" set "YTDLP=%LOCALAPPDATA%\Microsoft\WinGet\Links\yt-dlp.exe"
if not defined YTDLP if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages" (
  for /r "%LOCALAPPDATA%\Microsoft\WinGet\Packages" %%f in (yt-dlp.exe) do if not defined YTDLP if exist "%%f" set "YTDLP=%%f"
)
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
exit /b 0
