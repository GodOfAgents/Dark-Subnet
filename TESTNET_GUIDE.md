# 🧪 Dark Subnet - Testnet Setup Guide

This guide walks you through testing the Dark Subnet on Bittensor's testnet.

## Prerequisites

1. **Python 3.10+** installed
2. **Docker** (for Concrete ML on Windows)
3. **Testnet TAO** tokens (free from faucet)

---

## Step 1: Install Dependencies

```bash
# Install Bittensor
pip install bittensor

# Install project dependencies
pip install -r requirements.txt

# For full FHE (Linux/Docker only):
pip install concrete-ml
```

---

## Step 2: Create Wallets

You need two wallets: one for the miner and one for the validator.

```bash
# Create miner wallet
btcli wallet new_coldkey --wallet.name dark_miner
btcli wallet new_hotkey --wallet.name dark_miner --wallet.hotkey default

# Create validator wallet
btcli wallet new_coldkey --wallet.name dark_validator
btcli wallet new_hotkey --wallet.name dark_validator --wallet.hotkey default
```

**Save your mnemonics securely!**

---

## Step 3: Get Testnet TAO

Join the Bittensor Discord and request testnet TAO:
- Discord: https://discord.gg/bittensor
- Channel: `#testnet-faucet`

Check your balance:
```bash
btcli wallet balance --wallet.name dark_miner --subtensor.network test
btcli wallet balance --wallet.name dark_validator --subtensor.network test
```

---

## Step 4: Create or Join a Subnet

### Option A: Join an existing test subnet
```bash
# List available subnets
btcli subnet list --subtensor.network test

# Register on a subnet (costs ~1 TAO)
btcli subnet register \
    --netuid <SUBNET_ID> \
    --wallet.name dark_miner \
    --subtensor.network test
```

### Option B: Create your own subnet
```bash
# Create subnet (costs more TAO)
btcli subnet create \
    --wallet.name dark_validator \
    --subtensor.network test
```

---

## Step 5: Train the FHE Model

Before running neurons, train the FHE model:

```bash
# On Linux/Docker
python fhe_models/train_model.py

# On Windows (use Docker)
docker run -v ${PWD}:/app -w /app zamafhe/concrete-ml python fhe_models/train_model.py
```

---

## Step 6: Run the Miner

```bash
python neurons/miner.py \
    --netuid <YOUR_SUBNET_ID> \
    --wallet.name dark_miner \
    --wallet.hotkey default \
    --subtensor.network test \
    --logging.debug
```

The miner will:
1. Load the FHE model
2. Register its axon (endpoint) on the network
3. Wait for validator requests
4. Perform blind FHE inference

---

## Step 7: Run the Validator

In a **separate terminal**:

```bash
python neurons/validator.py \
    --netuid <YOUR_SUBNET_ID> \
    --wallet.name dark_validator \
    --wallet.hotkey default \
    --subtensor.network test \
    --logging.debug
```

The validator will:
1. Query registered miners
2. Send honey pot traps
3. Verify miner responses
4. Set weights on-chain

---

## Step 8: Monitor

### Check your registration
```bash
btcli subnet metagraph --netuid <SUBNET_ID> --subtensor.network test
```

### Check wallet balance
```bash
btcli wallet balance --wallet.name dark_miner --subtensor.network test
```

### View logs
Both neurons output detailed logs. Look for:
- `✓ Blind inference complete` (miner)
- `✓ Miner PASSED trap` (validator)

---

## Troubleshooting

### "Not registered"
Your wallet must be registered on the subnet:
```bash
btcli subnet register --netuid <ID> --wallet.name <NAME> --subtensor.network test
```

### "Insufficient stake"
Validators need stake. Add some:
```bash
btcli stake add --wallet.name dark_validator --subtensor.network test
```

### "FHE model not found"
Train the model first:
```bash
python fhe_models/train_model.py
```

### Connection issues
Check testnet status at: https://test.taostats.io

---

## Network Details

| Parameter | Value |
|-----------|-------|
| Network | `test` |
| Chain Endpoint | `wss://test.finney.opentensor.ai:443` |
| Block Time | ~12 seconds |
| Faucet | Discord #testnet-faucet |

---

## Quick Reference

```bash
# All commands use --subtensor.network test

# Register
btcli subnet register --netuid <ID> --wallet.name <NAME> --subtensor.network test

# Run miner
python neurons/miner.py --netuid <ID> --wallet.name dark_miner --subtensor.network test

# Run validator
python neurons/validator.py --netuid <ID> --wallet.name dark_validator --subtensor.network test

# Check metagraph
btcli subnet metagraph --netuid <ID> --subtensor.network test
```
