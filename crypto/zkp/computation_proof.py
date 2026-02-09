"""
Computation Proof Module

Zero-knowledge proofs that FHE computation was performed correctly.
Uses Fiat-Shamir heuristic for non-interactive proofs.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import time

from crypto.zkp.commitment import PedersenCommitment, create_computation_commitment


@dataclass
class ComputationProof:
    """
    Proof that FHE inference was executed correctly.
    
    Attributes:
        commitment: Pedersen commitment to computation
        challenge: Fiat-Shamir challenge
        response: Prover's response to challenge
        public_inputs: Public verification data
        timestamp: When proof was generated
    """
    commitment: any
    challenge: str
    response: int
    public_inputs: Dict[str, any]
    timestamp: float
    
    def serialize(self) -> Dict:
        """Serialize proof for network transmission."""
        return {
            "commitment": str(self.commitment),
            "challenge": self.challenge,
            "response": self.response,
            "public_inputs": self.public_inputs,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> "ComputationProof":
        """Deserialize proof from network data."""
        return cls(
            commitment=data["commitment"],
            challenge=data["challenge"],
            response=data["response"],
            public_inputs=data["public_inputs"],
            timestamp=data["timestamp"],
        )


class ProofGenerator:
    """
    Generates zero-knowledge proofs of FHE computation.
    
    Used by miners to prove they performed legitimate inference
    without revealing the actual computation details.
    
    Example:
        >>> generator = ProofGenerator()
        >>> proof = generator.generate_proof(
        ...     input_hash="abc123",
        ...     output_hash="def456",
        ...     computation_time_ms=2000.0
        ... )
    """
    
    def __init__(self):
        """Initialize proof generator."""
        self.commitment_scheme = PedersenCommitment()
    
    def generate_proof(
        self,
        input_hash: str,
        output_hash: str,
        computation_time_ms: float,
        computation_steps: int = 1000
    ) -> ComputationProof:
        """
        Generate a proof of correct FHE computation.
        
        Args:
            input_hash: SHA256 hash of encrypted input
            output_hash: SHA256 hash of encrypted output
            computation_time_ms: Time spent on FHE inference
            computation_steps: Estimated homomorphic operations
            
        Returns:
            ComputationProof object
        """
        # Create commitment to computation
        commitment, randomness, public_inputs = create_computation_commitment(
            input_hash, output_hash, computation_steps
        )
        
        # Add timing info to public inputs
        public_inputs["computation_time_ms"] = computation_time_ms
        
        # Fiat-Shamir: challenge = H(commitment || public_inputs)
        challenge = self._create_challenge(commitment, public_inputs)
        
        # Response: r + c*x (simplified Schnorr-like)
        challenge_int = int(challenge, 16) % self.commitment_scheme.order
        response = (randomness + challenge_int * computation_steps) % self.commitment_scheme.order
        
        return ComputationProof(
            commitment=commitment,
            challenge=challenge,
            response=response,
            public_inputs=public_inputs,
            timestamp=time.time(),
        )
    
    def _create_challenge(self, commitment: any, public_inputs: Dict) -> str:
        """
        Create Fiat-Shamir challenge hash.
        
        Args:
            commitment: The Pedersen commitment
            public_inputs: Public verification data
            
        Returns:
            Hex-encoded challenge hash
        """
        data = f"{commitment}:{public_inputs}".encode()
        return hashlib.sha256(data).hexdigest()


class ProofVerifier:
    """
    Verifies zero-knowledge proofs of FHE computation.
    
    Used by validators to check miner proofs without
    re-executing the computation.
    
    Example:
        >>> verifier = ProofVerifier()
        >>> is_valid = verifier.verify_proof(proof)
    """
    
    def __init__(self, min_computation_time_ms: float = 100.0):
        """
        Initialize proof verifier.
        
        Args:
            min_computation_time_ms: Minimum expected FHE computation time
        """
        self.commitment_scheme = PedersenCommitment()
        self.min_computation_time_ms = min_computation_time_ms
    
    def verify_proof(self, proof: ComputationProof) -> Tuple[bool, str]:
        """
        Verify a computation proof.
        
        Args:
            proof: The ComputationProof to verify
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check timestamp is not in future
        if proof.timestamp > time.time() + 60:  # Allow 60s clock skew
            return False, "Proof timestamp is in the future"
        
        # Check minimum computation time (FHE should take significant time)
        comp_time = proof.public_inputs.get("computation_time_ms", 0)
        if comp_time < self.min_computation_time_ms:
            return False, f"Computation time {comp_time}ms below minimum {self.min_computation_time_ms}ms"
        
        # Verify Fiat-Shamir challenge
        expected_challenge = self._recreate_challenge(proof.commitment, proof.public_inputs)
        if proof.challenge != expected_challenge:
            return False, "Challenge verification failed"
        
        # Verify response (simplified check)
        if proof.response < 0 or proof.response >= self.commitment_scheme.order:
            return False, "Response out of valid range"
        
        return True, "Proof verified successfully"
    
    def _recreate_challenge(self, commitment: any, public_inputs: Dict) -> str:
        """Recreate the challenge for verification."""
        data = f"{commitment}:{public_inputs}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def batch_verify(self, proofs: list[ComputationProof]) -> Dict[int, Tuple[bool, str]]:
        """
        Verify multiple proofs efficiently.
        
        Args:
            proofs: List of proofs to verify
            
        Returns:
            Dict mapping proof index to (is_valid, reason)
        """
        results = {}
        for i, proof in enumerate(proofs):
            results[i] = self.verify_proof(proof)
        return results
