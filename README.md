# 🌑 Dark Subnet - Privacy-First Bittensor Subnet

[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **"Computation on data you cannot see. Verification of work you cannot read."**

**Dark Subnet** is a pioneering Bittensor implementation that unlocks sensitive AI use cases (Healthcare, Finance, Privacy-Preserving GovTech) by combining **Fully Homomorphic Encryption (FHE)** with a novel **Honey Pot Verification** mechanism.

---

## 🔮 The Innovation

| Feature                | Standard Subnet                 | Dark Subnet                               |
|------------------------|---------------------------------|-------------------------------------------|
| **Data Visibility**    | 🔓 Public (Miners see raw data) | 🔒 **ZERO** (Miners see encrypted noise) |
| **Verification**       | ⚖️ Redundant (Multiple miners)  | 🍯 **Honey Pots** (Trap-based proof)     |
| **Privacy Compliance** | ❌ Risky (Data leaks)           | ✅ **HIPAA/GDPR** "Privacy by Design"    |
| **Incentive Layer**    | Token-based ranking             | Performance + Blind Accuracy              |

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

## 🧠 Proof of Blind Intelligence (PoBI)

Dark Subnet introduces a novel consensus mechanism: **Proof of Blind Intelligence** — proving legitimate AI computation without revealing data.

### Why It Works

| Property                         | Mechanism |
|----------------------------------|-----------|
| **Computational Irreducibility** | FHE operations cannot be shortcut — each inference requires O(n²) homomorphic multiplications |
| **Verifiable Effort**            | Honey pots prove miners did real work (correct trap = genuine FHE execution) |
| **Economic Alignment**           | Cost to cheat > Reward for cheating (stake at risk + 97% detection after 10 rounds) |

### Computation Cost Proof

| Operation               | Plaintext | FHE Ciphertext |
|-------------------------|-----------|----------------|
| Inference (10 features) | ~1ms      | ~2,000ms       |
| Memory footprint        | ~1KB      | ~50MB          |
| **Built-in PoW**        | ❌ None   | ✅ Inherent   |

> **Key Insight**: The 2000x computational overhead of FHE **IS** the proof of work. Lazy miners cannot fake it.

---

## 🔐 Cryptographic Security

Dark Subnet uses advanced cryptographic primitives for enhanced security:

| Component | Technology                   | Purpose                                        |
|-----------|------------------------------|------------------------------------------------|
| **ZKP**   | Pedersen Commitments (BN128) | Prove computation without revealing data       |
| **MPC**   | Shamir Secret Sharing        | Threshold key distribution (2-of-3)            |
| **FHE**   | Concrete-ML                  | Encrypted inference                            |

### Threshold Decryption
No single validator can decrypt honey pot results. Requires **k-of-n** validators to collaborate:
```
Master Key → [Share 1] + [Share 2] + [Share 3]
                  ↓            ↓
            Validator 1   Validator 2   → Reconstruct Key
```

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

## 💰 Tokenomics

**$DARK** powers the Dark Subnet economy. See [TOKENOMICS.md](TOKENOMICS.md) for:
- Emission schedule with halving
- Miner & Validator incentive design
- Anti-gaming mechanisms
- Proof of Blind Intelligence (PoBI) framework

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
│   └── validator.py      # Honey pot + crypto verification
├── protocol/
│   └── synapse.py        # FHESynapse & Batch definitions
├── crypto/
│   ├── zkp/              # Zero-Knowledge Proofs
│   │   ├── commitment.py # Pedersen commitments (BN128)
│   │   └── computation_proof.py  # Fiat-Shamir proofs
│   └── mpc/              # Multi-Party Computation
│       ├── shamir.py     # Secret sharing
│       └── threshold.py  # 2-of-3 decryption
├── fhe_models/
│   └── train_model.py    # Quantized FHE model training
├── client/
│   └── oracle.py         # Client encryption SDK
├── tests/                # 36 passing tests
├── scripts/              # Testnet automation tools
├── demo.py               # Main FHE demonstration script
├── demo_mock.py          # Lightweight simulator
├── Dockerfile            # Containerized environment
└── docker-compose.yml    # Miner-Validator orchestration
```

---

## ✅ Test Coverage

```bash
# Run all tests in Docker (recommended)
docker run dark-subnet python -m pytest tests/ -v

# Results: 36 passed
# - Crypto (ZKP/MPC): 26 tests
# - FHE Models: 6 tests
# - Synapse Protocol: 4 tests
```

---

## � Use Cases

Dark Subnet enables **any privacy-sensitive AI application**. The FHE + Honey Pot architecture is general-purpose:

### 💰 Finance
| Application | Privacy Benefit |
|-------------|-----------------|
| **Credit Scoring** | Assess risk without seeing income/debts |
| **Fraud Detection** | Analyze transactions without exposing amounts |
| **AML Compliance** | Check patterns on encrypted transaction data |
| **Insurance Underwriting** | Calculate premiums without medical records |

### 🏥 Healthcare
| Application | Privacy Benefit |
|-------------|-----------------|
| **Disease Risk Prediction** | Genetic risk without exposing DNA |
| **Drug Interaction Check** | Verify prescriptions without full history |
| **Clinical Trial Matching** | Match patients on encrypted conditions |
| **Mental Health Assessment** | Score wellbeing privately |

### 🏛️ Government / GovTech
| Application | Privacy Benefit |
|-------------|-----------------|
| **Benefits Eligibility** | Verify thresholds without exact amounts |
| **Tax Fraud Detection** | Pattern analysis on encrypted returns |
| **Voting Verification** | Validate eligibility anonymously |
| **Biometric Border Control** | Match on encrypted templates |

### 🤖 AI / Machine Learning
| Application | Privacy Benefit |
|-------------|-----------------|
| **Private LLM Inference** | Query AI on encrypted prompts |
| **Federated Learning** | Train on distributed encrypted data |
| **Recommendation Systems** | Personalize without exposing preferences |
| **Sentiment Analysis** | Analyze without reading messages |

### 🔐 Enterprise
| Application | Privacy Benefit |
|-------------|-----------------|
| **HR Analytics** | Salary benchmarking without individual pay |
| **Supply Chain** | Verify inventory without exposing pricing |
| **Legal Discovery** | Search documents without reading contents |
| **Competitive Intel** | Analyze encrypted market data |

---

## 🧪 Current Demo: Healthcare Credit Score

The included implementation demonstrates a **Logistic Regression** model trained on synthetic medical/financial data:
- **Input**: 10 encrypted features (payment history, income ratio, medical bills, etc.)
- **Output**: Binary risk classification (0=low, 1=high)
- **Privacy**: Server **never knows** the patient's age, BMI, or medical history

```bash
# Try it
python demo_mock.py           # Quick simulation
docker run dark-subnet python demo.py  # Full FHE
```

---

## 📜 License
MIT License - Built for Bittensor Hackathon 2026.
