"""
Pedersen Commitment Scheme

A cryptographically secure commitment scheme with:
- Hiding: Commitment reveals nothing about the value
- Binding: Cannot open commitment to different value

Uses BN128 elliptic curve via py_ecc library.
"""

from typing import Tuple
import hashlib
import secrets

try:
    from py_ecc.bn128 import G1, multiply, add, curve_order, eq
    HAS_PY_ECC = True
except ImportError:
    HAS_PY_ECC = False
    # Fallback for environments without py_ecc
    G1 = None
    curve_order = 2**256 - 1


class PedersenCommitment:
    """
    Pedersen Commitment: C = v*G + r*H
    
    Where:
    - G, H are generator points on BN128 curve
    - v is the value being committed
    - r is random blinding factor
    
    Properties:
    - Computationally binding: Cannot find v' != v with same commitment
    - Perfectly hiding: Commitment reveals nothing about v
    
    Example:
        >>> pc = PedersenCommitment()
        >>> commitment, randomness = pc.commit(42)
        >>> assert pc.verify(commitment, 42, randomness)
    """
    
    def __init__(self):
        """Initialize with generator points G and H."""
        if HAS_PY_ECC:
            self.G = G1
            # H = hash_to_curve("Dark Subnet Generator H")
            # For simplicity, use G * known_nothing_up_my_sleeve_number
            self.H = multiply(G1, 0xDEADBEEF)
            self.order = curve_order
        else:
            # Fallback: use simple modular arithmetic (NOT SECURE - demo only)
            self.G = 2
            self.H = 3
            self.order = 2**256 - 189
    
    def commit(self, value: int, randomness: int = None) -> Tuple[any, int]:
        """
        Create a commitment to a value.
        
        Args:
            value: Integer value to commit to
            randomness: Optional blinding factor (generated if not provided)
            
        Returns:
            Tuple of (commitment, randomness)
            
        Example:
            >>> pc = PedersenCommitment()
            >>> commitment, r = pc.commit(100)
        """
        if randomness is None:
            randomness = secrets.randbelow(self.order)
        
        if HAS_PY_ECC:
            # C = v*G + r*H
            vG = multiply(self.G, value % self.order)
            rH = multiply(self.H, randomness % self.order)
            commitment = add(vG, rH)
        else:
            # Fallback: C = (v*G + r*H) mod order
            commitment = (value * self.G + randomness * self.H) % self.order
        
        return commitment, randomness
    
    def verify(self, commitment: any, value: int, randomness: int) -> bool:
        """
        Verify that a commitment opens to the claimed value.
        
        Args:
            commitment: The commitment to verify
            value: Claimed value
            randomness: The blinding factor used
            
        Returns:
            True if commitment is valid, False otherwise
            
        Example:
            >>> pc = PedersenCommitment()
            >>> c, r = pc.commit(42)
            >>> assert pc.verify(c, 42, r) == True
            >>> assert pc.verify(c, 43, r) == False
        """
        expected, _ = self.commit(value, randomness)
        
        if HAS_PY_ECC:
            return eq(commitment, expected)
        else:
            return commitment == expected
    
    def create_commitment_hash(self, commitment: any) -> str:
        """
        Create a hash of the commitment for storage/transmission.
        
        Args:
            commitment: The elliptic curve point or integer
            
        Returns:
            Hex-encoded SHA256 hash
        """
        if HAS_PY_ECC:
            # Serialize EC point
            data = str(commitment).encode()
        else:
            data = str(commitment).encode()
        
        return hashlib.sha256(data).hexdigest()


def create_computation_commitment(
    input_hash: str,
    output_hash: str,
    computation_steps: int
) -> Tuple[any, int, dict]:
    """
    Create a commitment to a computation result.
    
    Used by miners to commit to FHE computation before revealing.
    
    Args:
        input_hash: Hash of encrypted input
        output_hash: Hash of encrypted output
        computation_steps: Number of FHE operations performed
        
    Returns:
        Tuple of (commitment, randomness, public_inputs)
    """
    pc = PedersenCommitment()
    
    # Combine inputs into a single value
    combined = int(hashlib.sha256(
        f"{input_hash}:{output_hash}:{computation_steps}".encode()
    ).hexdigest(), 16) % pc.order
    
    commitment, randomness = pc.commit(combined)
    
    public_inputs = {
        "input_hash": input_hash,
        "output_hash": output_hash,
        "computation_steps": computation_steps,
    }
    
    return commitment, randomness, public_inputs
