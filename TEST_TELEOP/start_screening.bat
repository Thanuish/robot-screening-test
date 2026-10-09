@echo off
rem Starts the remote screening test: the public tunnel in its own window (its https:// address
rem stays the same while that window is open, also when the lobby is restarted) and the lobby
rem (http://localhost:8000, results on /admin, which shows the public address).
rem Needs cloudflared:  winget install Cloudflare.cloudflared
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
where cloudflared >nul 2>nul || (
    echo cloudflared is not installed, so candidates can't reach this PC from the internet.
    echo Install it once with:   winget install Cloudflare.cloudflared
    echo then close this window and start again. Starting the lobby for local testing only...
    echo.
    goto lobby
)
tasklist /v /fi "imagename eq cloudflared.exe" | find /i "cloudflared" >nul || (
    start "Screening tunnel (keep open)" /min .venv\Scripts\python -m TEST_TELEOP.lobby tunnel
)
:lobby
start "" http://localhost:8000/admin
.venv\Scripts\python -m TEST_TELEOP.lobby serve %*
pause
