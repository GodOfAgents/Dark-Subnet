"""
FHESynapse - Protocol definition for encrypted data exchange.

This synapse carries FHE-encrypted data between validators and miners,
enabling blind inference where miners never see the plaintext.
"""

from typing import Optional
import bittensor as bt
from pydantic import Field


class FHESynapse(bt.Synapse):
    """
    Synapse for Fully Homomorphic Encryption (FHE) inference requests.
    
    The validator sends encrypted data to miners, who perform computation
    on the ciphertext and return encrypted results. Miners NEVER see
    the underlying plaintext data.
    
    Attributes:
        encrypted_data: FHE-encrypted input features (serialized bytes)
        evaluation_keys: Serialized FHE evaluation keys for computation
        request_id: Unique identifier for request tracking
        model_name: Target FHE model to use for inference
        is_honey_pot: Internal flag (never sent to miner) for trap requests
        encrypted_prediction: FHE-encrypted prediction result (filled by miner)
        computation_time_ms: Time taken for FHE inference (filled by miner)
    """
    
    # === REQUEST FIELDS (Validator -> Miner) ===
    
    # Encrypted input data - the miner sees only noise
    encrypted_data: bytes = Field(
        default=b"",
        title="Encrypted Data",
        description="FHE-encrypted input features. Miner cannot decrypt this.",
    )
    
    # Evaluation keys needed for FHE computation
    evaluation_keys: bytes = Field(
        default=b"",
        title="Evaluation Keys", 
        description="Serialized FHE evaluation keys for homomorphic operations.",
    )
    
    # Request metadata
    request_id: str = Field(
        default="",
        title="Request ID",
        description="Unique identifier for this inference request.",
    )
    
    model_name: str = Field(
        default="credit_scorer",
        title="Model Name",
        description="Name of the FHE model to use for inference.",
    )
    
    # === RESPONSE FIELDS (Miner -> Validator) ===
    
    # Encrypted prediction result
    encrypted_prediction: bytes = Field(
        default=b"",
        title="Encrypted Prediction",
        description="FHE-encrypted inference result. Only key holder can decrypt.",
    )
    
    # Performance metrics
    computation_time_ms: float = Field(
        default=0.0,
        title="Computation Time (ms)",
        description="Time taken to perform FHE inference in milliseconds.",
    )
    
    # Error handling
    error_message: Optional[str] = Field(
        default=None,
        title="Error Message",
        description="Error message if inference failed.",
    )
    
    # === INTERNAL FIELDS (Never serialized to network) ===
    
    # Honey pot flag - used by validator to track trap requests
    # This is NEVER sent to the miner
    _is_honey_pot: bool = False
    _expected_result: Optional[int] = None
    
    def deserialize(self) -> "FHESynapse":
        """Custom deserialization - ensures internal fields are not leaked."""
        return self
    
    class Config:
        """Pydantic configuration."""
        # Exclude internal fields from serialization
        fields = {
            "_is_honey_pot": {"exclude": True},
            "_expected_result": {"exclude": True},
        }


class BatchFHESynapse(bt.Synapse):
    """
    Synapse for batched FHE requests.
    
    Sends multiple encrypted requests in a single network call,
    mixing real requests with honey pot traps.
    """
    
    # List of encrypted data blobs
    encrypted_batch: list[bytes] = Field(
        default_factory=list,
        title="Encrypted Batch",
        description="List of FHE-encrypted inputs for batch processing.",
    )
    
    # Shared evaluation keys (same for all requests in batch)
    evaluation_keys: bytes = Field(
        default=b"",
        title="Evaluation Keys",
        description="Shared FHE evaluation keys for the batch.",
    )
    
    # Batch metadata
    batch_id: str = Field(
        default="",
        title="Batch ID",
        description="Unique identifier for this batch.",
    )
    
    model_name: str = Field(
        default="credit_scorer",
        title="Model Name",
        description="Target FHE model for inference.",
    )
    
    # Response: list of encrypted predictions
    encrypted_predictions: list[bytes] = Field(
        default_factory=list,
        title="Encrypted Predictions",
        description="List of FHE-encrypted inference results.",
    )
    
    # Total batch processing time
    total_computation_time_ms: float = Field(
        default=0.0,
        title="Total Computation Time (ms)",
        description="Total time to process the entire batch.",
    )
