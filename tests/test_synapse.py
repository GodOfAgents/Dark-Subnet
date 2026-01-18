"""
Unit tests for FHESynapse protocol.
"""

import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol.synapse import FHESynapse, BatchFHESynapse


class TestFHESynapse:
    """Tests for FHESynapse protocol definition."""
    
    def test_default_initialization(self):
        """Test default synapse creation."""
        synapse = FHESynapse()
        
        assert synapse.encrypted_data == b""
        assert synapse.evaluation_keys == b""
        assert synapse.request_id == ""
        assert synapse.model_name == "credit_scorer"
        assert synapse.encrypted_prediction == b""
        assert synapse.computation_time_ms == 0.0
        assert synapse.error_message is None
    
    def test_synapse_with_data(self):
        """Test synapse with encrypted data."""
        test_data = b"\x00\x01\x02\x03" * 100
        test_keys = b"\xff\xfe\xfd" * 50
        
        synapse = FHESynapse(
            encrypted_data=test_data,
            evaluation_keys=test_keys,
            request_id="test-123",
            model_name="credit_scorer",
        )
        
        assert synapse.encrypted_data == test_data
        assert synapse.evaluation_keys == test_keys
        assert synapse.request_id == "test-123"
        assert len(synapse.encrypted_data) == 400
    
    def test_internal_fields_excluded(self):
        """Test that internal honey pot fields are excluded from serialization."""
        synapse = FHESynapse()
        synapse._is_honey_pot = True
        synapse._expected_result = 1
        
        # Internal fields should not appear in dict representation
        data = synapse.model_dump()
        assert "_is_honey_pot" not in data
        assert "_expected_result" not in data
    
    def test_synapse_response_fields(self):
        """Test setting response fields."""
        synapse = FHESynapse()
        
        # Simulate miner response
        synapse.encrypted_prediction = b"\x00" * 100
        synapse.computation_time_ms = 150.5
        
        assert len(synapse.encrypted_prediction) == 100
        assert synapse.computation_time_ms == 150.5
    
    def test_synapse_error_handling(self):
        """Test error message field."""
        synapse = FHESynapse()
        synapse.error_message = "Model not found"
        
        assert synapse.error_message == "Model not found"


class TestBatchFHESynapse:
    """Tests for BatchFHESynapse protocol definition."""
    
    def test_default_batch_initialization(self):
        """Test default batch synapse creation."""
        synapse = BatchFHESynapse()
        
        assert synapse.encrypted_batch == []
        assert synapse.evaluation_keys == b""
        assert synapse.batch_id == ""
        assert synapse.encrypted_predictions == []
    
    def test_batch_with_multiple_requests(self):
        """Test batch with multiple encrypted requests."""
        batch = [
            b"\x00" * 100,
            b"\x01" * 100,
            b"\x02" * 100,
        ]
        
        synapse = BatchFHESynapse(
            encrypted_batch=batch,
            batch_id="batch-456",
            model_name="credit_scorer",
        )
        
        assert len(synapse.encrypted_batch) == 3
        assert synapse.batch_id == "batch-456"
    
    def test_batch_response(self):
        """Test batch response with predictions."""
        synapse = BatchFHESynapse()
        
        # Simulate batch processing
        synapse.encrypted_predictions = [
            b"\x00" * 50,
            b"\x01" * 50,
            b"\x02" * 50,
        ]
        synapse.total_computation_time_ms = 450.0
        
        assert len(synapse.encrypted_predictions) == 3
        assert synapse.total_computation_time_ms == 450.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
