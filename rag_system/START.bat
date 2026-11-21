@echo off
echo.
echo ============================================================
echo   Chat With My Documents - Local RAG System
echo ============================================================
echo.

REM Check if Ollama is running
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama is not running!
    echo [!] Please start Ollama first:
    echo     1. Open a new terminal
    echo     2. Run: ollama serve
    echo     3. Then run this script again
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running

REM Check if index exists
if not exist "chroma_db" (
    echo.
    echo [!] Document index not found. Running ingest first...
    echo.
    python ingest.py
    if %errorlevel% neq 0 (
        echo [ERROR] Ingest failed. Check the error above.
        pause
        exit /b 1
    )
)
echo [OK] Document index found

echo.
echo [*] Starting RAG web server...
echo [*] Open in browser: http://localhost:8000
echo [*] Press Ctrl+C to stop
echo.
python app.py
