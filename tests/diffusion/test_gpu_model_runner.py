# test_gpu_model_runner.py
import pytest
from unittest.mock import Mock
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

class TestOmniGPUModelRunnerAdditionalInformation:
    """测试 OmniGPUModelRunner._get_additional_information 方法"""
    
    def test_get_from_new_reqs(self):
        """从scheduled_new_reqs中获取additional_information"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        # Mock数据
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
        """从scheduled_cached_reqs中获取additional_information"""
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
        """从request_state的additional_information_cpu中获取"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        req_state = Mock()
        req_state.additional_information_cpu = {"last_talker_hidden": "hidden_state", "other": "value"}
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result == {"last_talker_hidden": "hidden_state", "other": "value"}
    
    def test_no_information_found(self):
        """找不到additional_information"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None
    
    def test_merge_cached_and_state_info(self):
        """合并缓存信息和request_state信息"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock()
        scheduler_output.scheduled_cached_reqs.additional_informations = {
            "req1": {"some_key": "some_value"}  # 缺少last_talker_hidden
        }
        
        req_state = Mock()
        req_state.additional_information_cpu = {
            "last_talker_hidden": "hidden_from_state",
            "num_processed_thinker_tokens": 10
        }
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        # 应该合并信息
        assert result == {
            "some_key": "some_value",
            "last_talker_hidden": "hidden_from_state",
            "num_processed_thinker_tokens": 10
        }
    
    def test_non_dict_info_becomes_none(self):
        """非字典类型的additional_information转换为None"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock()
        scheduler_output.scheduled_cached_reqs.additional_informations = {
            "req1": "not_a_dict"  # 不是字典
        }
        
        runner.requests = {}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None
    
    def test_req_state_info_not_dict(self):
        """request_state的additional_information_cpu不是字典"""
        runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
        
        scheduler_output = Mock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = Mock(additional_informations={})
        
        req_state = Mock()
        req_state.additional_information_cpu = "not_a_dict"  # 不是字典
        runner.requests = {"req1": req_state}
        
        result = runner._get_additional_information(scheduler_output, "req1")
        
        assert result is None