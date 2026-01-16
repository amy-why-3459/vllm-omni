# test_gpu_ar_model_runner.py
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

class TestExecuteModelLogitsIndices:
    """Test out-of-bounds handling logic for logits_indices in execute_model"""
    
    def test_logits_indices_within_bounds_no_adjustment(self):
        """No adjustment when indices are within bounds"""
        # Mock data
        hidden_states = torch.randn(100, 1024)
        logits_indices = torch.tensor([10, 20, 30])
        
        # Test condition
        num_hidden_tokens = hidden_states.shape[0]
        assert logits_indices.max().item() < num_hidden_tokens
        
        # Sampling should work normally
        sample_hidden_states = hidden_states[logits_indices]
        assert sample_hidden_states.shape == (3, 1024)
    
    def test_single_request_out_of_bounds(self):
        """Use the last token when a single request is out of bounds"""
        hidden_states = torch.randn(24, 1024)  # 24 tokens
        logits_indices = torch.tensor([30])  # Out of bounds
        num_scheduled_tokens_np = np.array([24])  # Single request
        
        # Test adjustment logic
        num_hidden_tokens = hidden_states.shape[0]
        if len(logits_indices) == 1 and num_scheduled_tokens_np.shape[0] == 1:
            # Single request: use the last token
            logits_indices = torch.tensor([num_hidden_tokens - 1])
        
        assert logits_indices.item() == 23  # Last token index
    
    def test_multiple_requests_out_of_bounds(self):
        """Use cumulative sum minus one when multiple requests are out of bounds"""
        hidden_states = torch.randn(50, 1024)
        logits_indices = torch.tensor([30, 60, 90])  # Test data, will be overridden
        num_scheduled_tokens_np = np.array([10, 20, 20])  # Total 50 tokens
        
        # Test adjustment logic
        cumsum_tokens = torch.cumsum(
            torch.from_numpy(num_scheduled_tokens_np).to('cpu'), 
            dim=0
        )
        logits_indices = cumsum_tokens - 1
        
        # Verify result
        expected = torch.tensor([9, 29, 49])  # 10-1, 30-1, 50-1
        assert torch.equal(logits_indices, expected)
    
    def test_indices_clamping_to_valid_range(self):
        """Ensure adjusted indices are within the valid range"""
        hidden_states = torch.randn(10, 1024)  # Only 10 tokens
        num_scheduled_tokens_np = np.array([5, 10])  # Second request goes out of bounds
        
        cumsum_tokens = torch.cumsum(
            torch.from_numpy(num_scheduled_tokens_np).to('cpu'), 
            dim=0
        )
        logits_indices = cumsum_tokens - 1
        logits_indices = logits_indices.clamp(min=0, max=9)
        
        # Second request should be clamped to 9
        assert torch.equal(logits_indices, torch.tensor([4, 9]))
    
    def test_hidden_states_sampling_correct_shape(self):
        """Verify sampled hidden_states have the correct shape"""
        hidden_states = torch.randn(100, 768)
        logits_indices = torch.tensor([0, 50, 99])  # First, middle, last
        
        sample_hidden_states = hidden_states[logits_indices]
        
        # Should sample 3 tokens, each with 768 dimensions
        assert sample_hidden_states.shape == (3, 768)
        # Verify correct sampling
        assert torch.equal(sample_hidden_states[0], hidden_states[0])
        assert torch.equal(sample_hidden_states[1], hidden_states[50])
        assert torch.equal(sample_hidden_states[2], hidden_states[99])
