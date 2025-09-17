@echo off
echo ========================================
echo   Advanced Semantic Analyzer v1.0
echo ========================================
echo.

cd /d "C:\Users\Joshua Ean\Downloads\datasets\DL_Sentiment_App"

echo Current directory: %CD%
echo.

echo Checking if model files exist...
if exist "best_llm_model.obj" (
    echo ✓ best_llm_model.obj found
) else (
    echo ✗ best_llm_model.obj NOT found
    echo Please ensure the model file is in the directory.
    pause
    exit /b 1
)

if exist "tokenizer.obj" (
    echo ✓ tokenizer.obj found
) else (
    echo ✗ tokenizer.obj NOT found
    echo Please ensure the tokenizer file is in the directory.
    pause
    exit /b 1
)

if exist "label_encoder.obj" (
    echo ✓ label_encoder.obj found
) else (
    echo ✗ label_encoder.obj NOT found
    echo Please ensure the label encoder file is in the directory.
    pause
    exit /b 1
)

if exist "model_metadata.obj" (
    echo ✓ model_metadata.obj found
) else (
    echo ✗ model_metadata.obj NOT found
    echo Please ensure the metadata file is in the directory.
    pause
    exit /b 1
)

echo.
echo All model files found! 🎉
echo.
echo Starting Advanced Semantic Analyzer...
echo You can access the web interface at: http://localhost:5001
echo Press Ctrl+C to stop the server.
echo.

"C:/Users/Joshua Ean/AppData/Local/Microsoft/WindowsApps/python3.13.exe" app.py

echo.
echo Semantic Analyzer has stopped.
pause
