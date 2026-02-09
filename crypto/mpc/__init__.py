"""
Multi-Party Computation Module

Provides threshold cryptography for distributed trust:
- ShamirSecretSharing: Split secrets into shares
- ThresholdDecryptor: Collaborative decryption requiring k-of-n validators
"""

from crypto.mpc.shamir import ShamirSecretSharing
from crypto.mpc.threshold import ThresholdDecryptor

__all__ = ["ShamirSecretSharing", "ThresholdDecryptor"]
