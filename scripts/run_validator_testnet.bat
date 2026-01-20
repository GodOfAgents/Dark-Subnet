@echo off
REM Dark Subnet - Run Validator on Testnet

set WALLET_NAME=dark_validator
set HOTKEY=default
set NETUID=1
set NETWORK=test

echo.
echo ==============================================
echo   Dark Subnet - Starting Validator (Testnet)
echo ==============================================
echo.
echo Wallet: %WALLET_NAME%
echo Network: %NETWORK%
echo Subnet: %NETUID%
echo.

python neurons/validator.py ^
    --netuid %NETUID% ^
    --wallet.name %WALLET_NAME% ^
    --wallet.hotkey %HOTKEY% ^
    --subtensor.network %NETWORK% ^
    --logging.debug

pause
