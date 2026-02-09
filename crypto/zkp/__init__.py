"""
Zero-Knowledge Proof Module

Provides cryptographic primitives for proving computation without revealing data:
- PedersenCommitment: Hiding and binding commitments
- ComputationProof: Prove FHE execution correctness
"""

from crypto.zkp.commitment import PedersenCommitment
from crypto.zkp.computation_proof import ComputationProof

__all__ = ["PedersenCommitment", "ComputationProof"]
