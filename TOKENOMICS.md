# 🌑 Dark Subnet Tokenomics

> **$DARK** — The native token powering privacy-preserving AI inference on Bittensor.

---

## 📊 Token Overview

| Property | Value |
|----------|-------|
| **Token Name** | Dark Token |
| **Symbol** | $DARK |
| **Network** | Bittensor Subnet |
| **Total Supply** | 21,000,000 DARK |
| **Initial Emission** | 1,000 DARK/epoch |

### Token Utility
- **Staking**: Miners and validators stake $DARK to participate
- **Rewards**: Earned for honest FHE computation and verification
- **Governance**: Future voting on subnet parameters
- **Fee Payment**: Clients pay $DARK for privacy-preserving inference

---

## 📈 Emission Schedule

### Halving Model
Emissions decrease every **10,000 blocks** (~35 days) to ensure long-term sustainability.

```
Emission(epoch) = INITIAL_EMISSION / 2^(epoch / HALVING_PERIOD)

Year 1: ~40% of total supply distributed
Year 2: ~20% distributed
Year 3+: Gradual decay toward zero
```

### Emission Curve

| Period | Blocks | DARK/Epoch | Cumulative % |
|--------|--------|------------|--------------|
| Era 1 | 0 - 10,000 | 1,000 | ~19% |
| Era 2 | 10,001 - 20,000 | 500 | ~29% |
| Era 3 | 20,001 - 30,000 | 250 | ~34% |
| Era 4 | 30,001 - 40,000 | 125 | ~37% |
| Era 5+ | 40,001+ | <62.5 | Asymptotic |

---

## ⛏️ Miner Incentives

Miners perform **blind FHE inference** on encrypted data. Rewards are calculated using a weighted score:

### Scoring Formula
```
MinerScore = (0.60 × HoneyPotAccuracy) + (0.25 × LatencyScore) + (0.15 × UptimeScore)
```

| Metric | Weight | Description |
|--------|--------|-------------|
| **Honey Pot Accuracy** | 60% | Correct responses on validator traps |
| **Latency Score** | 25% | FHE computation speed (lower = better) |
| **Uptime Score** | 15% | Availability over rolling 24h window |

### Reward Calculation
```
MinerReward = (EpochEmission × 0.70) × (MinerScore / TotalMinerScores)
```

### Penalty Mechanism
| Violation | Penalty |
|-----------|---------|
| Single honey pot failure | Score × 0.5 |
| 3 consecutive failures | Score × 0.1 |
| 10+ failures in 24h | Deregistration |

---

## ✅ Validator Incentives

Validators create **honey pot traps** and verify miner honesty blindly.

### Scoring Formula
```
ValidatorScore = (0.40 × TrapQuality) + (0.40 × ConsensusScore) + (0.20 × CoverageScore)
```

| Metric | Weight | Description |
|--------|--------|-------------|
| **Trap Quality** | 40% | Diversity and fairness of honey pots |
| **Consensus Score** | 40% | Agreement with other validators |
| **Coverage Score** | 20% | % of active miners verified per epoch |

### Reward Calculation
```
ValidatorReward = (EpochEmission × 0.30) × (ValidatorScore / TotalValidatorScores)
```

---

## 🛡️ Anti-Gaming Mechanisms

### Honey Pot Unpredictability
- **4 trap profile variations** (2 high-risk, 2 low-risk)
- Random selection per verification round
- Trap data is **indistinguishable** from real client data (FHE encryption)

### Trap Frequency
- ~30% of batch requests are traps
- Probabilistic detection: P(caught cheating in N rounds) = 1 - 0.7^N
- After 10 rounds: **97.2% detection probability**

### Economic Security
```
Cost to Cheat > Expected Gain from Cheating

- Stake at risk: Minimum 100 DARK
- Detection probability: ~30% per round
- Penalty on detection: Score halved + potential deregistration
```

---

## 🔐 Cryptographic Security Layer

Enhanced security through advanced cryptographic primitives:

### Zero-Knowledge Proofs (ZKP)
Miners generate **Pedersen commitments** to prove computation correctness:
```
Commitment(result) = result × G + randomness × H
```
- Validators verify proofs **without** re-executing FHE
- Invalid proofs → immediate score penalty

### Multi-Party Computation (MPC)
**Threshold decryption** prevents single-point-of-trust:
```
FHE Key → Split into 3 shares
Decryption requires: 2-of-3 validators
```
- No single validator can decrypt honey pot results
- Collusion resistance through Shamir Secret Sharing

### Security Guarantees
| Attack | Mitigation |
|--------|------------|
| Lazy computation | ZKP proves FHE was executed |
| Validator collusion | MPC threshold (k-of-n) |
| Honey pot prediction | FHE encryption indistinguishable |

---

## 🧠 Proof of Blind Intelligence (PoBI)

Dark Subnet introduces **Proof of Blind Intelligence** — a novel consensus mechanism proving legitimate AI computation without revealing data.

### Why FHE Proves Work
1. **Computational Irreducibility**: FHE operations cannot be shortcut
   - Each encrypted inference requires O(n²) homomorphic multiplications
   - No polynomial approximations possible on ciphertext

2. **Verifiable Effort**: Honey pots prove miners did real work
   - Correct trap results = genuine FHE execution
   - Wrong results = lazy/fake computation detected

3. **Economic Alignment**
   - Honest computation: Earn rewards
   - Cheating: Lose stake + get deregistered
   - Cost to fake > Reward for cheating

### Computation Cost Analysis
| Operation | Plaintext | FHE Ciphertext |
|-----------|-----------|----------------|
| Inference (10 features) | ~1ms | ~2,000ms |
| Memory footprint | ~1KB | ~50MB |
| **Proof of Work** | ❌ None | ✅ Built-in |

---

## 📋 Distribution Summary

| Allocation | Percentage | Purpose |
|------------|------------|---------|
| **Miner Rewards** | 70% | FHE inference computation |
| **Validator Rewards** | 30% | Honey pot verification |

### Future Considerations
- **Treasury**: Potential 5% allocation for subnet development
- **Burn Mechanism**: Fee burns for deflationary pressure
- **Governance**: Token-weighted voting on parameters

---

## 🔗 References

- [Validator Scoring Implementation](neurons/validator.py#L206-214)
- [FHE Synapse Protocol](protocol/synapse.py)
- [Bittensor Emission Docs](https://docs.bittensor.com)

---

*Last Updated: February 2026*
