@echo off
REM Dark Subnet - Testnet Setup Script for Windows
REM This script guides you through testnet setup

echo.
echo ==============================================
echo   Dark Subnet - Testnet Setup (Windows)
echo ==============================================
echo.

REM Configuration
set MINER_WALLET=dark_miner
set VALIDATOR_WALLET=dark_validator
set NETWORK=test
set NETUID=1

echo [Step 1] Checking bittensor installation...
pip show bittensor >nul 2>&1
if errorlevel 1 (
    echo Installing bittensor...
    pip install bittensor
) else (
    echo bittensor is already installed
)

echo.
echo [Step 2] Creating wallets...
echo.
echo Creating miner wallet: %MINER_WALLET%
btcli wallet new_coldkey --wallet.name %MINER_WALLET%
btcli wallet new_hotkey --wallet.name %MINER_WALLET% --wallet.hotkey default

echo.
echo Creating validator wallet: %VALIDATOR_WALLET%
btcli wallet new_coldkey --wallet.name %VALIDATOR_WALLET%
btcli wallet new_hotkey --wallet.name %VALIDATOR_WALLET% --wallet.hotkey default

echo.
echo [Step 3] Wallet addresses...
echo.
btcli wallet list

echo.
echo ==============================================
echo   Setup Complete!
echo ==============================================
echo.
echo Next steps:
echo.
echo 1. Get testnet TAO from Discord faucet:
echo    https://discord.gg/bittensor
echo    Channel: #testnet-faucet
echo.
echo 2. Register on subnet:
echo    btcli subnet register --netuid %NETUID% --wallet.name %MINER_WALLET% --subtensor.network %NETWORK%
echo    btcli subnet register --netuid %NETUID% --wallet.name %VALIDATOR_WALLET% --subtensor.network %NETWORK%
echo.
echo 3. Run neurons (after getting TAO and registering):
echo    python neurons/miner.py --netuid %NETUID% --wallet.name %MINER_WALLET% --subtensor.network %NETWORK%
echo    python neurons/validator.py --netuid %NETUID% --wallet.name %VALIDATOR_WALLET% --subtensor.network %NETWORK%
echo.
pause
