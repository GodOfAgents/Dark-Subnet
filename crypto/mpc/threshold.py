"""
Threshold Decryption Module

Enables collaborative decryption where k-of-n validators must
participate to decrypt FHE trap results.

This prevents any single validator from having full decryption power.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import uuid

from crypto.mpc.shamir import ShamirSecretSharing, split_fhe_key


class DecryptionStatus(Enum):
    """Status of a threshold decryption request."""
    PENDING = "pending"          # Waiting for shares
    READY = "ready"              # Has enough shares
    COMPLETED = "completed"      # Decryption finished
    FAILED = "failed"            # Not enough shares in time
    EXPIRED = "expired"          # Request timed out


@dataclass
class DecryptionRequest:
    """
    A request for threshold decryption.
    
    Attributes:
        request_id: Unique identifier
        ciphertext_hash: Hash of data to decrypt
        required_shares: Number of shares needed (k)
        total_validators: Total validators (n)
        collected_shares: Shares received so far
        status: Current request status
        created_at: Timestamp of request creation
        expires_at: When request expires
    """
    request_id: str
    ciphertext_hash: str
    required_shares: int
    total_validators: int
    collected_shares: Dict[int, bytes] = field(default_factory=dict)
    status: DecryptionStatus = DecryptionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 min timeout
    
    def add_share(self, validator_id: int, share: bytes) -> bool:
        """
        Add a share from a validator.
        
        Args:
            validator_id: ID of contributing validator
            share: The partial decryption share
            
        Returns:
            True if share was added successfully
        """
        if self.status not in [DecryptionStatus.PENDING, DecryptionStatus.READY]:
            return False
        
        if validator_id in self.collected_shares:
            return False  # Already have this validator's share
        
        self.collected_shares[validator_id] = share
        
        if len(self.collected_shares) >= self.required_shares:
            self.status = DecryptionStatus.READY
        
        return True
    
    def is_ready(self) -> bool:
        """Check if enough shares collected for decryption."""
        return len(self.collected_shares) >= self.required_shares
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() > self.expires_at


class ThresholdDecryptor:
    """
    Coordinates threshold decryption among validators.
    
    Ensures no single validator can decrypt honey pot results alone.
    Requires k-of-n validators to collaborate for decryption.
    
    Example:
        >>> decryptor = ThresholdDecryptor(n=3, k=2)
        >>> key_shares = decryptor.distribute_key(master_key)
        >>> request_id = decryptor.request_decryption(ciphertext_hash)
        >>> decryptor.contribute_share(request_id, validator_id=1, share=shares[0])
        >>> decryptor.contribute_share(request_id, validator_id=2, share=shares[1])
        >>> result = decryptor.try_decrypt(request_id)
    
    Attributes:
        n: Total number of validators
        k: Threshold for decryption
        requests: Active decryption requests
    """
    
    def __init__(self, n: int = 3, k: int = 2):
        """
        Initialize threshold decryptor.
        
        Args:
            n: Total validator count
            k: Minimum validators required for decryption
        """
        self.n = n
        self.k = k
        self.sss = ShamirSecretSharing(n=n, k=k)
        self.requests: Dict[str, DecryptionRequest] = {}
        self.key_shares: Dict[int, bytes] = {}  # validator_id -> key_share
    
    def distribute_key(self, master_key: bytes) -> Dict[int, bytes]:
        """
        Split master key among validators.
        
        Args:
            master_key: The FHE decryption key (16 bytes)
            
        Returns:
            Dict mapping validator_id to their key share
        """
        if len(master_key) != 16:
            # Pad or hash to 16 bytes
            master_key = hashlib.sha256(master_key).digest()[:16]
        
        shares = self.sss.split(master_key)
        
        self.key_shares = {idx: share for idx, share in shares}
        
        return self.key_shares
    
    def get_share_for_validator(self, validator_id: int) -> Optional[bytes]:
        """
        Get the key share for a specific validator.
        
        Args:
            validator_id: Validator's ID (1-indexed)
            
        Returns:
            Key share bytes or None if not found
        """
        return self.key_shares.get(validator_id)
    
    def request_decryption(self, ciphertext_hash: str) -> str:
        """
        Initiate a threshold decryption request.
        
        Args:
            ciphertext_hash: Hash of ciphertext to decrypt
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        self.requests[request_id] = DecryptionRequest(
            request_id=request_id,
            ciphertext_hash=ciphertext_hash,
            required_shares=self.k,
            total_validators=self.n,
        )
        
        return request_id
    
    def contribute_share(
        self,
        request_id: str,
        validator_id: int,
        share: bytes
    ) -> Tuple[bool, str]:
        """
        Contribute a partial decryption share.
        
        Args:
            request_id: The decryption request ID
            validator_id: Contributing validator's ID
            share: The partial decryption share
            
        Returns:
            Tuple of (success, message)
        """
        if request_id not in self.requests:
            return False, "Request not found"
        
        request = self.requests[request_id]
        
        if request.is_expired():
            request.status = DecryptionStatus.EXPIRED
            return False, "Request has expired"
        
        if not request.add_share(validator_id, share):
            return False, "Failed to add share (duplicate or invalid state)"
        
        shares_collected = len(request.collected_shares)
        shares_needed = request.required_shares
        
        if request.is_ready():
            return True, f"Ready for decryption ({shares_collected}/{shares_needed} shares)"
        
        return True, f"Share accepted ({shares_collected}/{shares_needed} shares)"
    
    def try_decrypt(self, request_id: str) -> Tuple[Optional[bytes], str]:
        """
        Attempt to complete threshold decryption.
        
        Args:
            request_id: The decryption request ID
            
        Returns:
            Tuple of (decrypted_key or None, status message)
        """
        if request_id not in self.requests:
            return None, "Request not found"
        
        request = self.requests[request_id]
        
        if request.is_expired():
            request.status = DecryptionStatus.EXPIRED
            return None, "Request has expired"
        
        if not request.is_ready():
            return None, f"Need {request.required_shares - len(request.collected_shares)} more shares"
        
        try:
            # Reconstruct key from shares
            shares = [(vid, share) for vid, share in request.collected_shares.items()]
            reconstructed = self.sss.combine(shares[:self.k])
            
            request.status = DecryptionStatus.COMPLETED
            
            return reconstructed, "Decryption successful"
            
        except Exception as e:
            request.status = DecryptionStatus.FAILED
            return None, f"Decryption failed: {str(e)}"
    
    def get_request_status(self, request_id: str) -> Optional[DecryptionRequest]:
        """Get the current status of a decryption request."""
        return self.requests.get(request_id)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired requests.
        
        Returns:
            Number of requests cleaned up
        """
        expired = [
            rid for rid, req in self.requests.items()
            if req.is_expired()
        ]
        
        for rid in expired:
            del self.requests[rid]
        
        return len(expired)
