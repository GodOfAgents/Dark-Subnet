# 🌑 Dark Subnet - Hackathon Pitch Deck

## The One-Liner
> **"We built the first Bittensor subnet where miners work on data they cannot see."**

---

## 🔮 The Problem

### Current Subnets Are Privacy-Blind

| What Miners See Today | The Risk |
|----------------------|----------|
| Medical records in text | HIPAA violations |
| Financial data | Fraud exposure |
| Personal prompts | Identity leaks |

**Every subnet miner today sees 100% of user data.**

---

## ⚡ Our Solution: Dark Subnet

### Fully Homomorphic Encryption (FHE) + Bittensor

```
┌─────────────┐    Encrypted    ┌─────────────┐    Encrypted    ┌─────────────┐
│   CLIENT    │ ──────────────> │    MINER    │ ──────────────> │   CLIENT    │
│  (Hospital) │   (Ciphertext)  │   (Blind)   │    (Result)     │  (Decrypt)  │
└─────────────┘                 └─────────────┘                 └─────────────┘
       │                               │                               │
       │  Patient data                 │  Sees ONLY noise              │  Only client
       │  (Age, BMI, History)          │  Cannot decrypt               │  can read result
       └───────────────────────────────┴───────────────────────────────┘
```

---

## 🎯 Three Hackathon Outcomes

### 1. Blind Inference ✅
```python
# Miner's forward function
def forward(self, synapse: FHESynapse) -> FHESynapse:
    # Miner NEVER sees age, income, or medical history
    encrypted_result = self.fhe_server.run(synapse.encrypted_data)
    return synapse
```

### 2. Blind Verification ✅
```python
# Validator's honey pot trap
trap_input = [99, 1, 1]  # Known "High Risk" profile
if miner_result != expected:
    score = 0.0  # Caught cheating!
```

### 3. Client Oracle ✅
```python
# Hospital decrypts and issues receipt
result = fhe_client.decrypt(encrypted_prediction)
receipt = sign("Request #505 processed correctly")
```

---

## 📊 Comparison Table

| Feature | Standard Subnet | Dark Subnet |
|---------|-----------------|-------------|
| **Data Visibility** | 🔓 Public | 🔒 **ZERO** |
| **Miner Knowledge** | Sees all text/images | Sees mathematical noise |
| **Verification** | Redundant (2 miners) | 🍯 Honey Pots |
| **Hardware** | GPU Required | ✅ CPU/GPU Agnostic |
| **Use Cases** | Chatbots | 🏥 Medical, 💰 Financial |
| **Compliance** | ❌ HIPAA/GDPR risk | ✅ Privacy by design |

---

## 🔐 The "Trust Sandwich" Protocol

### How We Verify Without Seeing

```
Step 1: Create TRAP with known answer
        ↓
Step 2: Mix trap with 9 real requests
        ↓
Step 3: Miner processes all 10 (can't distinguish)
        ↓
Step 4: Validator decrypts ONLY the trap
        ↓
Step 5: Trap correct? → Trust the other 9
```

**Result: Statistical verification without privacy leak**

---

## 🏗️ Technical Architecture

```
dark_subnet/
├── neurons/
│   ├── miner.py          # FHE inference (blind)
│   └── validator.py      # Honey pot verification
├── protocol/
│   └── synapse.py        # Encrypted data protocol
├── fhe_models/
│   └── train_model.py    # Concrete ML training
├── client/
│   └── oracle.py         # Client SDK
└── demo.py               # Live demonstration
```

### Tech Stack
- **FHE**: Zama Concrete ML
- **Network**: Bittensor
- **Model**: LogisticRegression (fast demo) / XGBoost (production)

---

## 🎬 Live Demo

### What You'll See

1. **Patient data encrypted** (client-side)
2. **Miner receives ciphertext** (only sees noise)
3. **Blind inference executes** (<1 second)
4. **Client decrypts result** (only client can read)
5. **Honey pot verification** (validator catches cheaters)

### Command
```bash
python demo.py
```

---

## 🚀 Why This Matters

### Unlocking $100B+ Markets

| Industry | Current Blocker | Dark Subnet Solution |
|----------|-----------------|---------------------|
| Healthcare | HIPAA compliance | FHE = data never exposed |
| Finance | Data sovereignty | Process without seeing |
| Insurance | Privacy regulations | Blind actuarial models |
| Government | Citizen privacy | Secure cloud compute |

---

## 👥 Team

- **Built at**: Bittensor Hackathon 2026
- **Tech**: Python, Zama Concrete ML, Bittensor SDK
- **Innovation**: First privacy-first subnet with FHE

---

## 📞 Call to Action

1. ⭐ **Star the repo**: github.com/dark-subnet
2. 🧪 **Try the demo**: `python demo.py`
3. 🤝 **Partner**: Medical AI, FinTech, GovTech

---

## 🏆 Why We Should Win

> "We didn't just build a subnet. We built **the future of private AI**."

- ✅ Novel cryptographic approach (FHE)
- ✅ Working demo (not vaporware)
- ✅ Real-world use cases (healthcare, finance)
- ✅ Scalable architecture (CPU/GPU agnostic)
- ✅ Proper verification (honey pots, not trust)

---

*"In the Dark Subnet, privacy isn't a feature. It's the foundation."*
