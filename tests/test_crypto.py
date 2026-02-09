"""
Comprehensive Test Suite for Crypto Modules

Tests for:
- ZKP: Pedersen commitments, computation proofs
- MPC: Shamir secret sharing, threshold decryption
"""

import pytest
import hashlib
import secrets


# ============================================================
# ZKP Tests
# ============================================================

class TestPedersenCommitment:
    """Tests for Pedersen commitment scheme."""
    
    def test_commit_creates_commitment(self):
        """Test that commit returns a commitment and randomness."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        commitment, randomness = pc.commit(42)
        
        assert commitment is not None
        assert randomness is not None
        assert isinstance(randomness, int)
    
    def test_verify_valid_commitment(self):
        """Test that valid commitments verify correctly."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        value = 12345
        commitment, randomness = pc.commit(value)
        
        assert pc.verify(commitment, value, randomness) is True
    
    def test_verify_rejects_wrong_value(self):
        """Test that wrong values fail verification."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        commitment, randomness = pc.commit(100)
        
        # Try to verify with wrong value
        assert pc.verify(commitment, 101, randomness) is False
    
    def test_verify_rejects_wrong_randomness(self):
        """Test that wrong randomness fails verification."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        commitment, randomness = pc.commit(100)
        
        # Try to verify with wrong randomness
        wrong_randomness = randomness + 1
        assert pc.verify(commitment, 100, wrong_randomness) is False
    
    def test_binding_property(self):
        """Test binding: same commitment requires same value."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        
        # Create two commitments with different values
        c1, r1 = pc.commit(42)
        c2, r2 = pc.commit(43)
        
        # They should be different
        assert c1 != c2
    
    def test_hiding_property(self):
        """Test hiding: commitment reveals nothing about value."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        
        # Same value with different randomness should give different commitments
        c1, r1 = pc.commit(42)
        c2, r2 = pc.commit(42)
        
        # With different randomness, commitments should differ
        assert r1 != r2  # Randomness should be different
    
    def test_commitment_hash(self):
        """Test commitment hash generation."""
        from crypto.zkp.commitment import PedersenCommitment
        
        pc = PedersenCommitment()
        commitment, _ = pc.commit(42)
        
        hash_result = pc.create_commitment_hash(commitment)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex


class TestComputationProof:
    """Tests for FHE computation proofs."""
    
    def test_generate_proof(self):
        """Test proof generation."""
        from crypto.zkp.computation_proof import ProofGenerator, ComputationProof
        
        generator = ProofGenerator()
        proof = generator.generate_proof(
            input_hash="abc123",
            output_hash="def456",
            computation_time_ms=2000.0
        )
        
        assert isinstance(proof, ComputationProof)
        assert proof.challenge is not None
        assert proof.timestamp > 0
    
    def test_verify_valid_proof(self):
        """Test that valid proofs verify."""
        from crypto.zkp.computation_proof import ProofGenerator, ProofVerifier
        
        generator = ProofGenerator()
        verifier = ProofVerifier(min_computation_time_ms=100.0)
        
        proof = generator.generate_proof(
            input_hash="abc123",
            output_hash="def456",
            computation_time_ms=2000.0
        )
        
        is_valid, message = verifier.verify_proof(proof)
        assert is_valid is True
    
    def test_reject_low_computation_time(self):
        """Test rejection of suspiciously fast computation."""
        from crypto.zkp.computation_proof import ProofGenerator, ProofVerifier
        
        generator = ProofGenerator()
        verifier = ProofVerifier(min_computation_time_ms=1000.0)
        
        # Computation too fast - suspicious!
        proof = generator.generate_proof(
            input_hash="abc123",
            output_hash="def456",
            computation_time_ms=50.0  # Way too fast for FHE
        )
        
        is_valid, message = verifier.verify_proof(proof)
        assert is_valid is False
        assert "below minimum" in message
    
    def test_proof_serialization(self):
        """Test proof serialization round-trip."""
        from crypto.zkp.computation_proof import ProofGenerator, ComputationProof
        
        generator = ProofGenerator()
        proof = generator.generate_proof(
            input_hash="test",
            output_hash="test2",
            computation_time_ms=1500.0
        )
        
        serialized = proof.serialize()
        deserialized = ComputationProof.deserialize(serialized)
        
        assert deserialized.challenge == proof.challenge
        assert deserialized.public_inputs == proof.public_inputs
    
    def test_batch_verify(self):
        """Test batch verification of multiple proofs."""
        from crypto.zkp.computation_proof import ProofGenerator, ProofVerifier
        
        generator = ProofGenerator()
        verifier = ProofVerifier(min_computation_time_ms=100.0)
        
        proofs = [
            generator.generate_proof(f"in{i}", f"out{i}", 2000.0)
            for i in range(5)
        ]
        
        results = verifier.batch_verify(proofs)
        
        assert len(results) == 5
        assert all(is_valid for is_valid, _ in results.values())


# ============================================================
# MPC Tests
# ============================================================

class TestShamirSSS:
    """Tests for Shamir Secret Sharing."""
    
    def test_split_creates_n_shares(self):
        """Test that split creates correct number of shares."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=5, k=3)
        secret = b"0123456789abcdef"  # 16 bytes
        
        shares = sss.split(secret)
        
        assert len(shares) == 5
    
    def test_combine_reconstructs_secret(self):
        """Test that k shares reconstruct the secret."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=5, k=3)
        secret = b"0123456789abcdef"
        
        shares = sss.split(secret)
        reconstructed = sss.combine(shares[:3])
        
        assert reconstructed == secret
    
    def test_threshold_reconstruction(self):
        """Test that exactly k shares are sufficient."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=5, k=3)
        secret = b"secretkey1234567"
        
        shares = sss.split(secret)
        
        # Any 3 shares should work
        assert sss.combine(shares[0:3]) == secret
        assert sss.combine(shares[1:4]) == secret
        assert sss.combine(shares[2:5]) == secret
    
    def test_insufficient_shares_fails(self):
        """Test that k-1 shares fail to reconstruct."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=5, k=3)
        secret = b"0123456789abcdef"
        
        shares = sss.split(secret)
        
        with pytest.raises(ValueError, match="Need at least"):
            sss.combine(shares[:2])  # Only 2 shares, need 3
    
    def test_wrong_secret_length_fails(self):
        """Test that non-16-byte secrets are rejected."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=3, k=2)
        
        with pytest.raises(ValueError, match="must be exactly 16 bytes"):
            sss.split(b"short")
    
    def test_invalid_threshold_fails(self):
        """Test that k > n is rejected."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        with pytest.raises(ValueError, match="cannot exceed"):
            ShamirSecretSharing(n=3, k=5)
    
    def test_verify_share_format(self):
        """Test share format validation."""
        from crypto.mpc.shamir import ShamirSecretSharing
        
        sss = ShamirSecretSharing(n=3, k=2)
        secret = b"0123456789abcdef"
        shares = sss.split(secret)
        
        # Valid share
        assert sss.verify_share(shares[0]) is True
        
        # Invalid shares
        assert sss.verify_share((0, b"test")) is False  # Invalid index
        assert sss.verify_share((1, b"short")) is False  # Wrong length


class TestThresholdDecryption:
    """Tests for threshold decryption coordinator."""
    
    def test_distribute_key(self):
        """Test key distribution creates shares for all validators."""
        from crypto.mpc.threshold import ThresholdDecryptor
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        master_key = b"masterkeymaster!"  # 16 bytes
        
        shares = decryptor.distribute_key(master_key)
        
        assert len(shares) == 3
        assert all(vid in shares for vid in [1, 2, 3])
    
    def test_request_decryption(self):
        """Test decryption request creation."""
        from crypto.mpc.threshold import ThresholdDecryptor, DecryptionStatus
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        
        request_id = decryptor.request_decryption("ciphertext_hash_abc")
        request = decryptor.get_request_status(request_id)
        
        assert request is not None
        assert request.status == DecryptionStatus.PENDING
    
    def test_contribute_share(self):
        """Test share contribution."""
        from crypto.mpc.threshold import ThresholdDecryptor
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        master_key = b"0123456789abcdef"
        shares = decryptor.distribute_key(master_key)
        
        request_id = decryptor.request_decryption("test_hash")
        
        success, msg = decryptor.contribute_share(request_id, 1, shares[1])
        assert success is True
    
    def test_threshold_decryption_flow(self):
        """Test complete threshold decryption workflow."""
        from crypto.mpc.threshold import ThresholdDecryptor, DecryptionStatus
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        master_key = b"0123456789abcdef"
        shares = decryptor.distribute_key(master_key)
        
        # Create request
        request_id = decryptor.request_decryption("test_cipher")
        
        # Contribute k shares
        decryptor.contribute_share(request_id, 1, shares[1])
        decryptor.contribute_share(request_id, 2, shares[2])
        
        # Should now be ready
        request = decryptor.get_request_status(request_id)
        assert request.status == DecryptionStatus.READY
        
        # Decrypt
        result, msg = decryptor.try_decrypt(request_id)
        
        assert result == master_key
        assert request.status == DecryptionStatus.COMPLETED
    
    def test_insufficient_shares_blocks_decrypt(self):
        """Test that decryption fails with insufficient shares."""
        from crypto.mpc.threshold import ThresholdDecryptor
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        master_key = b"0123456789abcdef"
        shares = decryptor.distribute_key(master_key)
        
        request_id = decryptor.request_decryption("test")
        
        # Only contribute 1 share (need 2)
        decryptor.contribute_share(request_id, 1, shares[1])
        
        result, msg = decryptor.try_decrypt(request_id)
        
        assert result is None
        assert "more shares" in msg
    
    def test_cleanup_expired(self):
        """Test cleanup of expired requests."""
        from crypto.mpc.threshold import ThresholdDecryptor
        import time
        
        decryptor = ThresholdDecryptor(n=3, k=2)
        
        # Create a request and manually expire it
        request_id = decryptor.request_decryption("test")
        decryptor.requests[request_id].expires_at = time.time() - 1
        
        cleaned = decryptor.cleanup_expired()
        
        assert cleaned == 1
        assert request_id not in decryptor.requests


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests combining ZKP and MPC."""
    
    def test_miner_proof_with_threshold_verification(self):
        """Test miner generating proof, validators verifying with MPC."""
        from crypto.zkp.computation_proof import ProofGenerator, ProofVerifier
        from crypto.mpc.threshold import ThresholdDecryptor
        
        # Miner generates proof
        generator = ProofGenerator()
        proof = generator.generate_proof(
            input_hash="encrypted_input_abc",
            output_hash="encrypted_output_xyz",
            computation_time_ms=2500.0
        )
        
        # Each validator verifies independently
        verifier = ProofVerifier()
        is_valid, _ = verifier.verify_proof(proof)
        
        assert is_valid is True
        
        # Validators use MPC for trap decryption
        decryptor = ThresholdDecryptor(n=3, k=2)
        trap_key = b"trapkey123456789"
        shares = decryptor.distribute_key(trap_key)
        
        # Simulate threshold decryption
        request_id = decryptor.request_decryption("trap_result_hash")
        decryptor.contribute_share(request_id, 1, shares[1])
        decryptor.contribute_share(request_id, 2, shares[2])
        
        recovered_key, _ = decryptor.try_decrypt(request_id)
        
        assert recovered_key == trap_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
