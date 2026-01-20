#!/bin/bash
# Dark Subnet - Testnet Setup Script
# This script automates testnet wallet creation and registration

set -e

echo "=============================================="
echo "  Dark Subnet - Testnet Setup"
echo "=============================================="
echo ""

# Configuration
MINER_WALLET="dark_miner"
VALIDATOR_WALLET="dark_validator"
NETWORK="test"
NETUID=${1:-1}  # Default to subnet 1, or pass as argument

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if btcli is installed
if ! command -v btcli &> /dev/null; then
    print_error "btcli not found. Installing bittensor..."
    pip install bittensor
fi

echo ""
print_step "Creating miner wallet: $MINER_WALLET"
if btcli wallet list 2>/dev/null | grep -q "$MINER_WALLET"; then
    print_warn "Wallet $MINER_WALLET already exists, skipping..."
else
    btcli wallet new_coldkey --wallet.name $MINER_WALLET --no-password
    btcli wallet new_hotkey --wallet.name $MINER_WALLET --wallet.hotkey default
fi

echo ""
print_step "Creating validator wallet: $VALIDATOR_WALLET"
if btcli wallet list 2>/dev/null | grep -q "$VALIDATOR_WALLET"; then
    print_warn "Wallet $VALIDATOR_WALLET already exists, skipping..."
else
    btcli wallet new_coldkey --wallet.name $VALIDATOR_WALLET --no-password
    btcli wallet new_hotkey --wallet.name $VALIDATOR_WALLET --wallet.hotkey default
fi

echo ""
print_step "Checking wallet balances..."
echo "Miner balance:"
btcli wallet balance --wallet.name $MINER_WALLET --subtensor.network $NETWORK 2>/dev/null || echo "  (no balance yet)"
echo ""
echo "Validator balance:"
btcli wallet balance --wallet.name $VALIDATOR_WALLET --subtensor.network $NETWORK 2>/dev/null || echo "  (no balance yet)"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Get testnet TAO from Discord faucet:"
echo "   https://discord.gg/bittensor"
echo "   Channel: #testnet-faucet"
echo ""
echo "2. Register on subnet $NETUID:"
echo "   btcli subnet register --netuid $NETUID --wallet.name $MINER_WALLET --subtensor.network $NETWORK"
echo "   btcli subnet register --netuid $NETUID --wallet.name $VALIDATOR_WALLET --subtensor.network $NETWORK"
echo ""
echo "3. Train FHE model:"
echo "   python fhe_models/train_model.py"
echo ""
echo "4. Run miner:"
echo "   python neurons/miner.py --netuid $NETUID --wallet.name $MINER_WALLET --subtensor.network $NETWORK"
echo ""
echo "5. Run validator (in another terminal):"
echo "   python neurons/validator.py --netuid $NETUID --wallet.name $VALIDATOR_WALLET --subtensor.network $NETWORK"
echo ""
