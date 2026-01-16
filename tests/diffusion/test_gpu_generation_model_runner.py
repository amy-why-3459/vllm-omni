# test_gpu_generation_model_runner.py
import pytest
from unittest.mock import Mock, patch
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

class TestGPUGenerationModelRunner:
    
    @patch.object(GPUGenerationModelRunner, '_init_mrope_positions')
    def test_update_request_states_basic(self, mock_init_mrope):
        """Test the basic workflow of updating request states"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        # Mock objects
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req1", "req2"]
        cached_reqs.prompt_token_ids = {"req1": [1, 2, 3], "req2": [4, 5, 6]}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        # Mock requests dictionary
        runner.requests = {}
        runner.input_batch = Mock()
        
        # Create mock request states
        for req_id in cached_reqs.req_ids:
            req_state = Mock()
            req_state.prompt_token_ids = None
            runner.requests[req_id] = req_state
        
        # Execute method
        runner._update_request_states(scheduler_output)
        
        # Verify results
        assert runner.requests["req1"].prompt_token_ids == [1, 2, 3]
        assert runner.requests["req2"].prompt_token_ids == [4, 5, 6]
        
        # Verify call counts
        assert runner.input_batch.remove_request.call_count == 2
        assert runner.input_batch.add_request.call_count == 2
        assert mock_init_mrope.call_count == 2
    
    def test_update_request_states_with_none_req_state(self):
        """Test the case where the request state does not exist"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req1"]
        cached_reqs.prompt_token_ids = {"req1": [1, 2, 3]}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        runner.requests = {}  # Empty dict, req1 does not exist
        
        # Should trigger an AssertionError
        with pytest.raises(AssertionError):
            runner._update_request_states(scheduler_output)
    
    @patch.object(GPUGenerationModelRunner, '_init_mrope_positions')
    def test_update_request_states_empty_cached_reqs(self, mock_init_mrope):
        """Test empty cached request list"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        cached_reqs = Mock()
        cached_reqs.req_ids = []  # Empty list
        cached_reqs.prompt_token_ids = {}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        runner.requests = {}
        runner.input_batch = Mock()
        
        # Execute method, should not raise errors
        runner._update_request_states(scheduler_output)
        
        # Verify no calls were made
        assert runner.input_batch.remove_request.call_count == 0
        assert runner.input_batch.add_request.call_count == 0
        assert mock_init_mrope.call_count == 0
