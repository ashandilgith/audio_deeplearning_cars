@echo off
REM Engine Ear (cars side project) - step 1b: extra search YouTube for car engine fault sound videos.
REM Double-click. Writes car_search_results2.txt next to this file (titles and ids only, no downloads).
setlocal
cd /d "%~dp0"
call :find_ytdlp
if not defined YTDLP (
  echo yt-dlp not found. Installing it with winget...
  winget install -e --id yt-dlp.yt-dlp --accept-source-agreements --accept-package-agreements
  call :find_ytdlp
)
if not defined YTDLP ( echo Could not find yt-dlp. & pause & exit /b 1 )
if exist car_search_results2.txt del car_search_results2.txt
for /f "usebackq tokens=1,2 delims=|" %%a in ("car_queries2.txt") do (
  echo Searching: %%b
  echo ### %%a^|%%b>> car_search_results2.txt
  "%YTDLP%" "ytsearch25:%%b" --flat-playlist --ignore-errors --print "%%(id)s	%%(duration)s	%%(channel)s	%%(title)s" >> car_search_results2.txt
)
echo.
echo Done. Tell Claude the search has finished.
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
