# test_gpu_generation_model_runner.py
import pytest
from unittest.mock import Mock, patch
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

class TestGPUGenerationModelRunner:
    
    @patch.object(GPUGenerationModelRunner, '_init_mrope_positions')
    def test_update_request_states_basic(self, mock_init_mrope):
        """测试更新请求状态的基本流程"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        # Mock对象
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req1", "req2"]
        cached_reqs.prompt_token_ids = {"req1": [1, 2, 3], "req2": [4, 5, 6]}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        # Mock requests字典
        runner.requests = {}
        runner.input_batch = Mock()
        
        # 创建请求状态mock
        for req_id in cached_reqs.req_ids:
            req_state = Mock()
            req_state.prompt_token_ids = None
            runner.requests[req_id] = req_state
        
        # 执行方法
        runner._update_request_states(scheduler_output)
        
        # 验证
        assert runner.requests["req1"].prompt_token_ids == [1, 2, 3]
        assert runner.requests["req2"].prompt_token_ids == [4, 5, 6]
        
        # 验证调用顺序
        assert runner.input_batch.remove_request.call_count == 2
        assert runner.input_batch.add_request.call_count == 2
        assert mock_init_mrope.call_count == 2
    
    def test_update_request_states_with_none_req_state(self):
        """测试请求状态不存在的情况"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req1"]
        cached_reqs.prompt_token_ids = {"req1": [1, 2, 3]}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        runner.requests = {}  # 空字典，req1不存在
        
        # 应该触发assert断言
        with pytest.raises(AssertionError):
            runner._update_request_states(scheduler_output)
    
    @patch.object(GPUGenerationModelRunner, '_init_mrope_positions')
    def test_update_request_states_empty_cached_reqs(self, mock_init_mrope):
        """测试空缓存请求列表"""
        runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
        
        cached_reqs = Mock()
        cached_reqs.req_ids = []  # 空列表
        cached_reqs.prompt_token_ids = {}
        
        scheduler_output = Mock()
        scheduler_output.scheduled_cached_reqs = cached_reqs
        
        runner.requests = {}
        runner.input_batch = Mock()
        
        # 执行方法，应该不报错
        runner._update_request_states(scheduler_output)
        
        # 验证没有调用
        assert runner.input_batch.remove_request.call_count == 0
        assert runner.input_batch.add_request.call_count == 0
        assert mock_init_mrope.call_count == 0