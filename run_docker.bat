@echo off
REM Dark Subnet - Docker Runner for Windows
REM 
REM This script runs the FHE demo using Docker (recommended for Windows)
REM
REM Prerequisites:
REM   - Docker Desktop installed and running
REM   - WSL2 backend enabled (recommended)

echo.
echo ============================================================
echo   Dark Subnet - FHE Demo Runner
echo   Running with Docker (Concrete ML compatible)
echo ============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check which command to run
if "%1"=="" goto demo
if "%1"=="demo" goto demo
if "%1"=="train" goto train
if "%1"=="miner" goto miner
if "%1"=="validator" goto validator
if "%1"=="simulation" goto simulation
if "%1"=="shell" goto shell
if "%1"=="build" goto build

:usage
echo Usage: run_docker.bat [command]
echo.
echo Commands:
echo   demo        Run the full demo (default)
echo   train       Train the FHE model
echo   miner       Run simulated miner
echo   validator   Run simulated validator
echo   simulation  Run miner + validator together
echo   shell       Open a shell in the container
echo   build       Rebuild the Docker image
echo.
goto end

:build
echo [*] Building Docker image...
docker build -t dark-subnet .
echo [OK] Build complete
goto end

:train
echo [*] Training FHE model...
docker run --rm -v "%cd%\fhe_models:/app/fhe_models" dark-subnet python fhe_models/train_model.py
goto end

:demo
echo [*] Running full demo...
docker run --rm -it -v "%cd%\fhe_models:/app/fhe_models" dark-subnet python demo.py
goto end

:miner
echo [*] Starting simulated miner on port 8091...
docker run --rm -it -p 8091:8091 -v "%cd%\fhe_models:/app/fhe_models" dark-subnet python neurons/miner.py --simulate
goto end

:validator
echo [*] Starting simulated validator...
docker run --rm -it -v "%cd%\fhe_models:/app/fhe_models" dark-subnet python neurons/validator.py --simulate --rounds 5
goto end

:simulation
echo [*] Running miner + validator simulation with docker-compose...
docker-compose up
goto end

:shell
echo [*] Opening shell in container...
docker run --rm -it -v "%cd%:/app" dark-subnet bash
goto end

:end
echo.
echo Done!
