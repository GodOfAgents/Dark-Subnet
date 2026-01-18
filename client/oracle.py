"""
Client Oracle SDK - Privacy-Preserving Inference for End Clients

This module provides client-side encryption and verification capabilities
for end clients (hospitals, banks, etc.) who want to use the Dark Subnet.

The Workflow:
1. Client encrypts their sensitive data locally
2. Encrypted data is sent to the subnet (via validator)
3. Miners perform blind inference (never see plaintext)
4. Client receives encrypted result
5. Client decrypts result locally and issues a "receipt"
6. Receipt can be used to boost miner weights (external proof)

Key Innovation: The client is the ONLY entity that ever sees the data.
Neither the miner nor the validator can decrypt it.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import base64

import numpy as np
from rich.console import Console

console = Console()


@dataclass
class InferenceReceipt:
    """
    Cryptographic receipt proving correct inference.
    
    This receipt can be sent back to the subnet to boost miner weights,
    providing external proof of quality from the actual data owner.
    """
    request_id: str
    timestamp: float
    result_hash: str
    decryption_success: bool
    client_signature: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize to JSON for network transmission."""
        return json.dumps({
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "result_hash": self.result_hash,
            "decryption_success": self.decryption_success,
            "client_signature": self.client_signature,
        })
    
    @classmethod
    def from_json(cls, data: str) -> "InferenceReceipt":
        """Deserialize from JSON."""
        obj = json.loads(data)
        return cls(**obj)


class BaseClient:
    """
    Base client for privacy-preserving FHE inference.
    
    This client handles all cryptographic operations locally,
    ensuring that sensitive data never leaves the client's control.
    """
    
    def __init__(self, model_path: Path, client_id: Optional[str] = None):
        """
        Initialize the FHE client.
        
        Args:
            model_path: Path to the compiled FHE model directory
            client_id: Optional client identifier for receipts
        """
        from concrete.ml.deployment import FHEModelClient
        
        self.model_path = model_path
        self.client_id = client_id or str(uuid.uuid4())[:8]
        
        # Load FHE client
        self.fhe_client = FHEModelClient(str(model_path))
        self.evaluation_keys = self.fhe_client.get_serialized_evaluation_keys()
        
        console.print(f"[green]✓ FHE Client initialized (ID: {self.client_id})[/green]")
    
    def encrypt(self, features: np.ndarray) -> tuple[bytes, str]:
        """
        Encrypt input features for blind inference.
        
        Args:
            features: Input features as numpy array (1 x n_features)
            
        Returns:
            Tuple of (encrypted_data, request_id)
        """
        request_id = f"{self.client_id}-{uuid.uuid4()}"
        
        # Quantize and encrypt
        encrypted = self.fhe_client.quantize_encrypt_serialize(features)
        
        console.print(f"[cyan]Encrypted {features.shape} -> {len(encrypted)} bytes[/cyan]")
        
        return encrypted, request_id
    
    def decrypt(self, encrypted_result: bytes) -> np.ndarray:
        """
        Decrypt inference result.
        
        Args:
            encrypted_result: FHE-encrypted prediction
            
        Returns:
            Decrypted prediction as numpy array
        """
        result = self.fhe_client.deserialize_decrypt_dequantize(encrypted_result)
        
        console.print(f"[green]Decrypted result: {result}[/green]")
        
        return result
    
    def create_receipt(
        self,
        request_id: str,
        encrypted_result: bytes,
        decryption_success: bool,
    ) -> InferenceReceipt:
        """
        Create a cryptographic receipt for the inference.
        
        This receipt proves that:
        1. The client received a valid encrypted result
        2. The result was successfully decrypted
        3. The client vouches for the miner's work
        
        Args:
            request_id: The original request identifier
            encrypted_result: The encrypted prediction received
            decryption_success: Whether decryption succeeded
            
        Returns:
            InferenceReceipt that can be sent to boost miner weight
        """
        result_hash = hashlib.sha256(encrypted_result).hexdigest()
        
        receipt = InferenceReceipt(
            request_id=request_id,
            timestamp=time.time(),
            result_hash=result_hash,
            decryption_success=decryption_success,
        )
        
        # In production, sign with client's private key
        # For demo, we just hash the receipt content
        receipt_content = f"{request_id}:{result_hash}:{decryption_success}"
        receipt.client_signature = hashlib.sha256(receipt_content.encode()).hexdigest()[:16]
        
        return receipt


class HealthcareClient(BaseClient):
    """
    Client SDK for healthcare applications.
    
    Example use cases:
    - Patient risk assessment
    - Disease prediction
    - Treatment outcome prediction
    - Insurance claim analysis
    """
    
    FEATURE_NAMES = [
        "age_normalized",          # Age scaled to [0, 1]
        "bmi_normalized",          # BMI scaled to [0, 1]
        "blood_pressure_systolic", # Normalized BP
        "blood_pressure_diastolic",
        "glucose_level",           # Normalized glucose
        "insulin_level",           # Normalized insulin
        "family_history",          # 0 or 1
        "smoking_status",          # 0 or 1
        "physical_activity",       # Normalized score
        "previous_conditions",     # Count normalized
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize healthcare client."""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "fhe_models" / "credit_scorer"
        
        super().__init__(model_path, client_id="HEALTH")
    
    def encrypt_patient_data(
        self,
        age: float,
        bmi: float,
        bp_systolic: float,
        bp_diastolic: float,
        glucose: float,
        insulin: float,
        family_history: bool,
        smoker: bool,
        activity_level: float,
        prev_conditions: int,
    ) -> tuple[bytes, str]:
        """
        Encrypt patient data for blind inference.
        
        All values should be normalized to [0, 1] range.
        
        Returns:
            Tuple of (encrypted_data, request_id)
        """
        features = np.array([[
            age,
            bmi,
            bp_systolic,
            bp_diastolic,
            glucose,
            insulin,
            1.0 if family_history else 0.0,
            1.0 if smoker else 0.0,
            activity_level,
            min(prev_conditions / 5.0, 1.0),  # Normalize to [0, 1]
        ]], dtype=np.float32)
        
        return self.encrypt(features)
    
    def decrypt_risk_assessment(self, encrypted_result: bytes) -> dict:
        """
        Decrypt and interpret a risk assessment result.
        
        Returns:
            Dictionary with risk level and recommendation
        """
        result = self.decrypt(encrypted_result)
        risk_class = int(result[0])
        
        if risk_class == 1:
            return {
                "risk_level": "HIGH",
                "recommendation": "Recommend immediate follow-up with specialist",
                "action": "URGENT",
            }
        else:
            return {
                "risk_level": "LOW",
                "recommendation": "Continue routine monitoring",
                "action": "ROUTINE",
            }


class FinancialClient(BaseClient):
    """
    Client SDK for financial applications.
    
    Example use cases:
    - Credit scoring
    - Fraud detection
    - Loan approval prediction
    - Investment risk assessment
    """
    
    FEATURE_NAMES = [
        "age_normalized",
        "annual_income",
        "debt_to_income",
        "num_credit_accounts",
        "payment_history_score",
        "credit_utilization",
        "years_credit_history",
        "num_hard_inquiries",
        "has_mortgage",
        "has_default_history",
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize financial client."""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "fhe_models" / "credit_scorer"
        
        super().__init__(model_path, client_id="FINANCE")
    
    def encrypt_credit_application(
        self,
        age: float,
        annual_income: float,
        debt_ratio: float,
        num_accounts: int,
        payment_score: float,
        utilization: float,
        credit_years: float,
        inquiries: int,
        has_mortgage: bool,
        has_defaults: bool,
    ) -> tuple[bytes, str]:
        """
        Encrypt credit application data for blind scoring.
        
        All values should be normalized to [0, 1] range.
        
        Returns:
            Tuple of (encrypted_data, request_id)
        """
        features = np.array([[
            age,
            annual_income,
            debt_ratio,
            min(num_accounts / 10.0, 1.0),
            payment_score,
            utilization,
            min(credit_years / 30.0, 1.0),
            min(inquiries / 10.0, 1.0),
            1.0 if has_mortgage else 0.0,
            1.0 if has_defaults else 0.0,
        ]], dtype=np.float32)
        
        return self.encrypt(features)
    
    def decrypt_credit_decision(self, encrypted_result: bytes) -> dict:
        """
        Decrypt and interpret a credit decision.
        
        Returns:
            Dictionary with decision and explanation
        """
        result = self.decrypt(encrypted_result)
        risk_class = int(result[0])
        
        if risk_class == 1:
            return {
                "decision": "DECLINED",
                "risk_level": "HIGH",
                "explanation": "Application exceeds risk threshold",
            }
        else:
            return {
                "decision": "APPROVED",
                "risk_level": "LOW",
                "explanation": "Application meets credit criteria",
            }


# ============================================================
# Demo Functions
# ============================================================

def demo_healthcare_workflow():
    """Demonstrate healthcare client workflow."""
    console.print("\n[bold cyan]=== Healthcare Client Demo ===[/bold cyan]\n")
    
    try:
        client = HealthcareClient()
        
        # Encrypt patient data
        console.print("[yellow]Encrypting patient data...[/yellow]")
        encrypted, request_id = client.encrypt_patient_data(
            age=0.65,           # 65 years old
            bmi=0.7,            # High BMI
            bp_systolic=0.8,    # Elevated BP
            bp_diastolic=0.75,
            glucose=0.6,
            insulin=0.5,
            family_history=True,
            smoker=True,
            activity_level=0.2,  # Low activity
            prev_conditions=2,
        )
        
        console.print(f"[green]✓ Request ID: {request_id}[/green]")
        console.print(f"[green]✓ Encrypted size: {len(encrypted)} bytes[/green]")
        console.print("[cyan]→ This encrypted data can be sent to miners[/cyan]")
        console.print("[cyan]→ Miners will compute without seeing the patient info[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Run 'python fhe_models/train_model.py' first[/yellow]")


def demo_financial_workflow():
    """Demonstrate financial client workflow."""
    console.print("\n[bold cyan]=== Financial Client Demo ===[/bold cyan]\n")
    
    try:
        client = FinancialClient()
        
        # Encrypt credit application
        console.print("[yellow]Encrypting credit application...[/yellow]")
        encrypted, request_id = client.encrypt_credit_application(
            age=0.35,
            annual_income=0.75,
            debt_ratio=0.25,
            num_accounts=5,
            payment_score=0.9,
            utilization=0.3,
            credit_years=12,
            inquiries=1,
            has_mortgage=True,
            has_defaults=False,
        )
        
        console.print(f"[green]✓ Request ID: {request_id}[/green]")
        console.print(f"[green]✓ Encrypted size: {len(encrypted)} bytes[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Run 'python fhe_models/train_model.py' first[/yellow]")


if __name__ == "__main__":
    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║   Dark Subnet - Client Oracle SDK        ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
    
    demo_healthcare_workflow()
    demo_financial_workflow()
