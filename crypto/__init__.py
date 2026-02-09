"""
Dark Subnet Cryptographic Module

Advanced cryptographic primitives for secure computation:
- ZKP: Zero-Knowledge Proofs (Pedersen commitments, computation proofs)
- MPC: Multi-Party Computation (Shamir SSS, threshold decryption)
"""

from crypto.zkp import PedersenCommitment, ComputationProof
from crypto.mpc import ShamirSecretSharing, ThresholdDecryptor

__all__ = [
    "PedersenCommitment",
    "ComputationProof", 
    "ShamirSecretSharing",
    "ThresholdDecryptor",
]
