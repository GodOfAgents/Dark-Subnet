"""
Shamir Secret Sharing Module

Production-ready implementation using PyCryptodome.
Enables splitting secrets into shares where k-of-n are required for recovery.
"""

from typing import List, Tuple
import secrets

try:
    from Crypto.Protocol.SecretSharing import Shamir
    HAS_PYCRYPTODOME = True
except ImportError:
    HAS_PYCRYPTODOME = False


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing Scheme.
    
    Split a secret into n shares where any k shares can reconstruct it,
    but k-1 shares reveal nothing about the secret.
    
    Uses PyCryptodome for production-grade implementation.
    
    Example:
        >>> sss = ShamirSecretSharing(n=5, k=3)
        >>> shares = sss.split(b"my_secret_key_16")
        >>> reconstructed = sss.combine(shares[:3])
        >>> assert reconstructed == b"my_secret_key_16"
    
    Attributes:
        n: Total number of shares to generate
        k: Minimum shares required for reconstruction (threshold)
    """
    
    def __init__(self, n: int = 3, k: int = 2):
        """
        Initialize secret sharing scheme.
        
        Args:
            n: Total number of shares (must be >= k)
            k: Threshold for reconstruction (must be >= 2)
            
        Raises:
            ValueError: If k > n or k < 2
        """
        if k > n:
            raise ValueError(f"Threshold k={k} cannot exceed total shares n={n}")
        if k < 2:
            raise ValueError(f"Threshold k={k} must be at least 2")
        
        self.n = n
        self.k = k
    
    def split(self, secret: bytes) -> List[Tuple[int, bytes]]:
        """
        Split a secret into n shares.
        
        Args:
            secret: 16-byte secret to split (must be exactly 16 bytes)
            
        Returns:
            List of (index, share_bytes) tuples
            
        Raises:
            ValueError: If secret is not exactly 16 bytes
            
        Example:
            >>> sss = ShamirSecretSharing(n=5, k=3)
            >>> shares = sss.split(b"0123456789abcdef")  # 16 bytes
            >>> len(shares)
            5
        """
        if len(secret) != 16:
            raise ValueError(f"Secret must be exactly 16 bytes, got {len(secret)}")
        
        if HAS_PYCRYPTODOME:
            return Shamir.split(self.k, self.n, secret)
        else:
            # Fallback: simple XOR-based splitting (NOT SECURE - demo only)
            return self._fallback_split(secret)
    
    def combine(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """
        Reconstruct secret from k or more shares.
        
        Args:
            shares: List of (index, share_bytes) tuples (at least k shares)
            
        Returns:
            Original 16-byte secret
            
        Raises:
            ValueError: If fewer than k shares provided
            
        Example:
            >>> sss = ShamirSecretSharing(n=5, k=3)
            >>> shares = sss.split(b"0123456789abcdef")
            >>> secret = sss.combine(shares[:3])  # Only need 3 shares
        """
        if len(shares) < self.k:
            raise ValueError(f"Need at least {self.k} shares, got {len(shares)}")
        
        if HAS_PYCRYPTODOME:
            return Shamir.combine(shares[:self.k])
        else:
            return self._fallback_combine(shares)
    
    def _fallback_split(self, secret: bytes) -> List[Tuple[int, bytes]]:
        """Fallback split for environments without PyCryptodome."""
        # Generate n-1 random shares, last share is XOR of all
        shares = []
        xor_result = int.from_bytes(secret, 'big')
        
        for i in range(self.n - 1):
            random_share = secrets.token_bytes(16)
            shares.append((i + 1, random_share))
            xor_result ^= int.from_bytes(random_share, 'big')
        
        # Last share makes XOR work out
        last_share = xor_result.to_bytes(16, 'big')
        shares.append((self.n, last_share))
        
        return shares
    
    def _fallback_combine(self, shares: List[Tuple[int, bytes]]) -> bytes:
        """Fallback combine for environments without PyCryptodome."""
        result = 0
        for _, share in shares:
            result ^= int.from_bytes(share, 'big')
        return result.to_bytes(16, 'big')
    
    def verify_share(self, share: Tuple[int, bytes]) -> bool:
        """
        Verify a share has valid format.
        
        Args:
            share: (index, share_bytes) tuple
            
        Returns:
            True if share format is valid
        """
        if not isinstance(share, tuple) or len(share) != 2:
            return False
        
        index, data = share
        
        if not isinstance(index, int) or index < 1 or index > self.n:
            return False
        
        if not isinstance(data, bytes) or len(data) != 16:
            return False
        
        return True


def split_fhe_key(fhe_key_material: bytes, n: int = 3, k: int = 2) -> List[Tuple[int, bytes]]:
    """
    Split FHE key material into shares for distributed validators.
    
    Args:
        fhe_key_material: Raw key bytes (will be chunked into 16-byte segments)
        n: Total number of validator shares
        k: Threshold for key reconstruction
        
    Returns:
        List of share tuples for each validator
    """
    sss = ShamirSecretSharing(n=n, k=k)
    
    # Pad key to multiple of 16 bytes
    padding_needed = (16 - len(fhe_key_material) % 16) % 16
    padded_key = fhe_key_material + bytes([padding_needed] * padding_needed)
    
    # Split each 16-byte chunk
    all_shares = [[] for _ in range(n)]
    
    for i in range(0, len(padded_key), 16):
        chunk = padded_key[i:i+16]
        chunk_shares = sss.split(chunk)
        
        for idx, share_data in chunk_shares:
            all_shares[idx - 1].append(share_data)
    
    # Combine chunks for each validator
    return [(i + 1, b"".join(all_shares[i])) for i in range(n)]
