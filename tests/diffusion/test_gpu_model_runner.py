# test_gpu_model_runner.py
import pytest
from unittest.mock import Mock
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

class TestOmniGPUModelRunnerAdditionalInformation:
    """Tests for OmniGPUModelRunner._get_additional_information method"""
    
    def test_get_from_new_reqs(self):
        """Get additional_information from scheduled_new_reqs"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        # Mock data
        new_req = Mock()
        new_req.req_id = "req1"
        new_req.additional_information = {"key": "value_from_new"}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = [new_req]
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result == {"key": "value_from_new"}
    
    def test_get_from_cached_reqs(self):
        """Get additional_information from scheduled_cached_reqs"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock()
        scheduler_output.scheduled_cached_reqs.additional_informations = {
            "req1": {"key": "value_from_cache"}
        }
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result == {"key": "value_from_cache"}
    
    def test_get_from_request_state(self):
        """Get additional_information from request_state.additional_information_cpu"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        req_state = Mock()
        req_state.additional_information_cpu = {
            "last_talker_hidden": "hidden_state",
            "other": "value"
        }
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result == {
            "last_talker_hidden": "hidden_state",
            "other": "value"
        }
    
    def test_no_information_found(self):
        """Return None when no additional_information is found"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None
    
    def test_merge_cached_and_state_info(self):
        """Merge cached information and request_state information"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock()
        scheduler_output.scheduled_cached_reqs.additional_informations = {
            "req1": {"some_key": "some_value"}  # Missing last_talker_hidden
        }
        
        req_state = Mock()
        req_state.additional_information_cpu = {
            "last_talker_hidden": "hidden_from_state",
            "num_processed_thinker_tokens": 10
        }
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        # Information should be merged
        assert result == {
            "some_key": "some_value",
            "last_talker_hidden": "hidden_from_state",
            "num_processed_thinker_tokens": 10
        }
    
    def test_non_dict_info_becomes_none(self):
        """Convert non-dict additional_information to None"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock()
        scheduler_output.scheduled_cached_reqs.additional_informations = {
            "req1": "not_a_dict"  # Not a dict
        }
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None
    
    def test_req_state_info_not_dict(self):
        """request_state.additional_information_cpu is not a dict"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        req_state = Mock()
        req_state.additional_information_cpu = "not_a_dict"  # Not a dict
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None
