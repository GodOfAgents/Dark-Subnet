# 🌑 Dark Subnet - Privacy-First Bittensor Subnet with FHE

A hackathon demo demonstrating **Blind Inference** and **Blind Verification** using Fully Homomorphic Encryption (FHE).

## 🔮 The Innovation

| Feature | Standard Subnet | Dark Subnet |
|---------|-----------------|-------------|
| Data Visibility | Public | **Zero** (miners see noise) |
| Verification | Redundant | **Honey Pots** (traps) |
| Hardware | GPU Dependent | **CPU/GPU Agnostic** |
| Use Case | Chatbots | **Medical/Financial** |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended for Windows)

```bash
# Build the image
docker build -t dark-subnet .

# Run the demo
docker run -it dark-subnet python demo.py

# Or use the batch script
run_docker.bat demo
```

### Option 2: Docker Compose (Full Simulation)

```bash
# Train model + run miner + validator
docker-compose up

# Or run separately:
docker-compose up train    # Train FHE model
docker-compose up miner    # Start miner on port 8091
docker-compose up validator # Start validator
```

### Option 3: Native Python (Linux/WSL2)

```bash
pip install -r requirements.txt
python demo.py
```

### Option 4: Mock Demo (Windows - No Docker)

```bash
pip install numpy scikit-learn rich requests
python demo_mock.py
```

---

## 🐳 Docker Commands

| Command | Description |
|---------|-------------|
| `run_docker.bat demo` | Run full FHE demo |
| `run_docker.bat train` | Train FHE model |
| `run_docker.bat miner` | Start simulated miner |
| `run_docker.bat validator` | Start simulated validator |
| `run_docker.bat simulation` | Run miner + validator together |
| `run_docker.bat shell` | Open bash in container |

---

## 📁 Project Structure

```
dark_subnet/
├── neurons/
│   ├── miner.py          # Blind FHE inference server
│   └── validator.py      # Honey pot verification
├── protocol/
│   └── synapse.py        # FHESynapse definition
├── fhe_models/
│   ├── train_model.py    # Train & compile FHE circuit
│   └── credit_scorer/    # Compiled model artifacts
├── client/
│   └── oracle.py         # Client-side encryption SDK
├── demo.py               # Full FHE demo
├── demo_mock.py          # Mock demo (Windows)
├── run_simulation.py     # Miner+Validator sim
├── Dockerfile            # Docker build
└── docker-compose.yml    # Multi-container setup
```

---

## 🔐 How It Works

### 1. Blind Inference (Miner)
```python
# Miner receives encrypted data
encrypted_result = fhe_server.run(encrypted_input)
# Miner NEVER sees: age, income, medical history
```

### 2. Blind Verification (Validator)
```python
# Validator creates trap with known output
trap = encrypt([age=99, smoker=yes])  # Known "High Risk"
if miner_result != expected:
    score = 0.0  # Caught cheating!
```

### 3. The Trust Sandwich Protocol
```
Validator → Creates TRAP (known output)
    ↓
Trap is ENCRYPTED → Miner can't tell it's a trap
    ↓
Miner processes → Returns encrypted result
    ↓
Validator DECRYPTS trap → Verifies correctness
    ↓
If correct → Miner is trusted
```

---

## 🎬 Running Modes

| Mode | Command | FHE | Use Case |
|------|---------|-----|----------|
| Docker Demo | `run_docker.bat demo` | ✅ Real | Production demo |
| Docker Sim | `docker-compose up` | ✅ Real | Miner+Validator |
| Mock Demo | `python demo_mock.py` | ❌ Simulated | Windows quick test |
| Native | `python demo.py` | ✅ Real | Linux/WSL2 |

---

## ⚠️ Platform Notes

### Windows
Concrete ML requires Linux. Use one of:
- **Docker Desktop** (recommended)
- **WSL2** with Ubuntu
- **Mock demo** (`demo_mock.py`)

### Linux / macOS
```bash
pip install concrete-ml
python demo.py
```

---

## 📊 Hackathon Summary

| What We Built | Technology |
|---------------|------------|
| Blind Inference | Zama Concrete ML (FHE) |
| Blind Verification | Honey Pot Traps |
| Network Protocol | Bittensor Subnet |
| Use Cases | Healthcare, Finance |

**Key Innovation**: Miners work on data they cannot see. Validators grade work without seeing answers.

---

## 📜 License

MIT License - Built for Bittensor Hackathon 2026
