# 🌑 Dark Subnet - Privacy-First Bittensor Subnet

> **"Computation on data you cannot see. Verification of work you cannot read."**

**Dark Subnet** is a pioneering Bittensor implementation that unlocks sensitive AI use cases (Healthcare, Finance, Privacy-Preserving GovTech) by combining **Fully Homomorphic Encryption (FHE)** with a novel **Honey Pot Verification** mechanism.

---

## 🔮 The Innovation

| Feature | Standard Subnet | Dark Subnet |
|---------|-----------------|-------------|
| **Data Visibility** | 🔓 Public (Miners see raw data) | 🔒 **ZERO** (Miners see encrypted noise) |
| **Verification** | ⚖️ Redundant (Multiple miners) | 🍯 **Honey Pots** (Trap-based proof) |
| **Privacy Compliance** | ❌ Risky (Data leaks) | ✅ **HIPAA/GDPR** "Privacy by Design" |
| **Incentive Layer** | Token-based ranking | Performance + Blind Accuracy |

---

## ⚡ Core Architecture

### 1. Blind Inference (Miner)
Miners function as "blind executors." They receive FHE-encrypted ciphertext and compute results using `concrete-ml` without ever decrypting the input.
- **Input**: Encrypted mathematical noise.
- **Output**: Encrypted result, decryptable only by the Client.

### 2. Blind Verification (Validator)
The **"Trust Sandwich"** protocol:
1. Validator generates a **Trap** (Honey Pot) with a known correct answer.
2. The Trap is encrypted and mixed with real Client requests.
3. Miner processes both without being able to distinguish them.
4. Validator decrypts **only** the Trap result to score the Miner’s honesty.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Docker Desktop** (Essential for Windows/macOS to run Linux-only FHE libraries)

### Installation
```bash
git clone https://github.com/GodOfAgents/Dark-Subnet.git
cd Dark-Subnet
pip install -r requirements.txt
```

---

## 🎮 Running the Demo

### Option 1: Full FHE Demo (Windows/Docker)
Run the complete end-to-end flow (Model training → Encryption → Blind Inference → Decryption):
```bash
# Using the Windows helper script
run_docker.bat demo

# OR manually
docker build -t dark-subnet .
docker run -it dark-subnet python demo.py
```

### Option 2: Mock Demo (Instant / No Docker)
For a quick visual overview of the concept without heavy FHE libraries:
```bash
python demo_mock.py
```

---

## 🧪 Testnet Participation

We have provided automated scripts to help you join the Bittensor Testnet.

1. **Setup Wallets**: `scripts/setup_testnet.bat` (Creates coldkeys/hotkeys)
2. **Register**: Use `btcli subnet register --netuid 1 --subtensor.network test`
3. **Launch Miner**: `scripts/run_miner_testnet.bat`
4. **Launch Validator**: `scripts/run_validator_testnet.bat`

*Check [TESTNET_GUIDE.md](TESTNET_GUIDE.md) for detailed step-by-step instructions.*

---

## 📁 Repository Structure

```
Dark-Subnet/
├── neurons/
│   ├── miner.py          # Blind FHE inference neuron
│   └── validator.py      # Honey pot scoring neuron
├── protocol/
│   └── synapse.py        # FHESynapse & Batch definitions
├── fhe_models/
│   └── train_model.py    # Quantized FHE model training
├── client/
│   └── oracle.py         # Client encryption SDK
├── scripts/              # Testnet automation tools
├── demo.py               # Main FHE demonstration script
├── demo_mock.py          # Lightweight simulator
├── Dockerfile            # Containerized environment
└── docker-compose.yml    # Miner-Validator orchestration
```

---

## 🏥 Use Case: Healthcare Credit Score
The current implementation uses a **Logistic Regression** model trained on synthetic medical/financial data. 
- **Goal**: Predict credit risk or health outcomes for patients.
- **Privacy Result**: The server computing the risk **never knows** the patient's age, BMI, or medical history.

## 📜 License
MIT License - Built for Bittensor Hackathon 2026.
