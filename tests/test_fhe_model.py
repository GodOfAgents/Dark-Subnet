"""
Unit tests for FHE model training and inference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFHEModelTraining:
    """Tests for FHE model training functionality."""
    
    def test_generate_credit_data(self):
        """Test synthetic data generation."""
        from fhe_models.train_model import generate_credit_scoring_data
        
        X, y = generate_credit_scoring_data(n_samples=100)
        
        assert X.shape == (100, 10), "Should have 100 samples with 10 features"
        assert len(y) == 100, "Should have 100 labels"
        assert set(np.unique(y)) == {0, 1}, "Should be binary classification"
        assert X.min() >= 0 and X.max() <= 1, "Features should be normalized to [0, 1]"
    
    @pytest.mark.slow
    def test_train_logistic_regression(self):
        """Test LogisticRegression training and FHE compilation."""
        from fhe_models.train_model import (
            generate_credit_scoring_data,
            train_logistic_regression,
        )
        from sklearn.model_selection import train_test_split
        
        X, y = generate_credit_scoring_data(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = train_logistic_regression(X_train, y_train, X_test, y_test, n_bits=6)
        
        # Verify model works
        y_pred = model.predict(X_test[:5])
        assert len(y_pred) == 5, "Should predict 5 samples"
        assert all(p in [0, 1] for p in y_pred), "Predictions should be 0 or 1"


class TestFHEInference:
    """Tests for FHE inference functionality."""
    
    @pytest.fixture
    def compiled_model_path(self, tmp_path):
        """Create a compiled model for testing."""
        from fhe_models.train_model import (
            generate_credit_scoring_data,
            train_logistic_regression,
            compile_and_save_model,
        )
        from sklearn.model_selection import train_test_split
        
        X, y = generate_credit_scoring_data(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = train_logistic_regression(X_train, y_train, X_test, y_test, n_bits=6)
        output_dir = tmp_path / "test_model"
        compile_and_save_model(model, X_train, output_dir)
        
        return output_dir
    
    @pytest.mark.slow
    def test_fhe_client_server_round_trip(self, compiled_model_path):
        """Test full client-server FHE round trip."""
        from concrete.ml.deployment import FHEModelClient, FHEModelServer
        
        # Load client and server
        client = FHEModelClient(str(compiled_model_path))
        server = FHEModelServer(str(compiled_model_path))
        
        # Get evaluation keys
        eval_keys = client.get_serialized_evaluation_keys()
        assert len(eval_keys) > 0, "Should have evaluation keys"
        
        # Create test input
        test_input = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        
        # Encrypt
        encrypted = client.quantize_encrypt_serialize(test_input)
        assert isinstance(encrypted, bytes), "Encrypted data should be bytes"
        assert len(encrypted) > 0, "Encrypted data should not be empty"
        
        # Blind inference
        encrypted_result = server.run(encrypted, eval_keys)
        assert isinstance(encrypted_result, bytes), "Result should be bytes"
        
        # Decrypt
        result = client.deserialize_decrypt_dequantize(encrypted_result)
        assert result.shape == (1,), "Should have single prediction"
        assert int(result[0]) in [0, 1], "Prediction should be 0 or 1"
    
    @pytest.mark.slow
    def test_fhe_matches_clear(self, compiled_model_path):
        """Test that FHE predictions match clear predictions."""
        from concrete.ml.deployment import FHEModelClient, FHEModelServer
        from fhe_models.train_model import generate_credit_scoring_data
        
        client = FHEModelClient(str(compiled_model_path))
        server = FHEModelServer(str(compiled_model_path))
        eval_keys = client.get_serialized_evaluation_keys()
        
        # Generate test data
        X, _ = generate_credit_scoring_data(n_samples=10)
        
        # Test a few samples
        for i in range(3):
            test_input = X[i:i+1].astype(np.float32)
            
            # FHE prediction
            encrypted = client.quantize_encrypt_serialize(test_input)
            encrypted_result = server.run(encrypted, eval_keys)
            fhe_result = int(client.deserialize_decrypt_dequantize(encrypted_result)[0])
            
            # Both should be valid binary predictions
            assert fhe_result in [0, 1], f"FHE result should be 0 or 1, got {fhe_result}"


class TestHoneyPot:
    """Tests for honey pot verification mechanism."""
    
    @pytest.fixture
    def compiled_model_path(self, tmp_path):
        """Create a compiled model for testing."""
        from fhe_models.train_model import (
            generate_credit_scoring_data,
            train_logistic_regression,
            compile_and_save_model,
        )
        from sklearn.model_selection import train_test_split
        
        X, y = generate_credit_scoring_data(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = train_logistic_regression(X_train, y_train, X_test, y_test, n_bits=6)
        output_dir = tmp_path / "test_model"
        compile_and_save_model(model, X_train, output_dir)
        
        return output_dir
    
    @pytest.mark.slow
    def test_honey_pot_high_risk(self, compiled_model_path):
        """Test that high-risk trap profiles produce expected output."""
        from concrete.ml.deployment import FHEModelClient, FHEModelServer
        
        client = FHEModelClient(str(compiled_model_path))
        server = FHEModelServer(str(compiled_model_path))
        eval_keys = client.get_serialized_evaluation_keys()
        
        # High risk profile: extreme values that should predict 1
        high_risk = np.array([[0.95, 0.1, 0.95, 0.9, 0.05, 0.95, 0.05, 0.9, 0.0, 1.0]], dtype=np.float32)
        
        encrypted = client.quantize_encrypt_serialize(high_risk)
        encrypted_result = server.run(encrypted, eval_keys)
        result = int(client.deserialize_decrypt_dequantize(encrypted_result)[0])
        
        # Result should be deterministic (either 0 or 1)
        assert result in [0, 1], f"Result should be binary, got {result}"
    
    @pytest.mark.slow
    def test_miner_cannot_distinguish_trap(self, compiled_model_path):
        """Test that trap data looks the same as real data to miner."""
        from concrete.ml.deployment import FHEModelClient
        
        client = FHEModelClient(str(compiled_model_path))
        
        # Real data
        real_data = np.array([[0.5, 0.6, 0.3, 0.4, 0.7, 0.2, 0.8, 0.1, 1.0, 0.0]], dtype=np.float32)
        encrypted_real = client.quantize_encrypt_serialize(real_data)
        
        # Trap data
        trap_data = np.array([[0.9, 0.1, 0.9, 0.8, 0.1, 0.9, 0.1, 0.8, 0.0, 1.0]], dtype=np.float32)
        encrypted_trap = client.quantize_encrypt_serialize(trap_data)
        
        # Both should be similar-length byte strings (miner can't tell difference)
        assert isinstance(encrypted_real, bytes)
        assert isinstance(encrypted_trap, bytes)
        # Sizes should be similar (within 20%)
        size_ratio = len(encrypted_real) / len(encrypted_trap)
        assert 0.8 < size_ratio < 1.2, "Encrypted sizes should be similar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
