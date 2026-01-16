import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from vllm.v1.request import RequestStatus

import sys
import os

from vllm_omni.distributed.omni_connectors.adapter import (
    get_chunk,
    get_through_connector,
    validate_talker_output,
    get_chunk_for_generation,
    put_chunk
)


class TestOmniConnectorAdapter:
    """测试 OmniConnector 适配器函数"""
    
    @pytest.fixture
    def mock_connector(self):
        """创建模拟的 connector"""
        connector = Mock()
        connector.stage_id = 1
        connector.get_requests = {}
        connector.finished_requests = set()
        connector.put_requests = {}
        connector.request_prompt_token_ids = {}
        return connector
    
    @pytest.fixture
    def mock_scheduler_output(self):
        """创建模拟的 scheduler_output"""
        output = Mock()
        
        # 模拟新的请求
        new_req = Mock()
        new_req.req_id = "req_1"
        new_req.additional_information = None
        output.scheduled_new_reqs = [new_req]
        
        # 模拟缓存的请求
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req_2", "req_3"]
        cached_reqs.additional_informations = {}
        output.scheduled_cached_reqs = cached_reqs
        
        return output

    @pytest.fixture
    def mock_request(self):
        """创建模拟的 request"""
        request = Mock()
        request.request_id = "req_1"
        request.prompt_token_ids = []
        request.status = None
        return request
    
    def test_get_chunk_for_stage_0(self, mock_connector, mock_scheduler_output):
        """测试 stage_id=0 时 get_chunk 直接返回"""
        # 设置 stage_id=0
        mock_connector.stage_id = 0
        
        # 调用 get_chunk
        result = get_chunk(mock_connector, mock_scheduler_output)
        
        # stage_id=0 应该直接返回，不执行后续逻辑
        assert result is None
    
    def test_get_chunk_new_requests(self, mock_connector, mock_scheduler_output):
        """测试 get_chunk 处理新请求"""
        # 设置测试数据
        mock_connector.get_requests = {"req_1": 0}
        
        # 【关键修复】确保缓存请求列表为空，避免干扰对新请求的测试
        mock_scheduler_output.scheduled_cached_reqs.req_ids = []  # 清空缓存请求列表
        
        # 模拟 get_through_connector 返回有效数据
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": False
        }

        with patch('vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
                return_value=mock_payload):
            # 调用 get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # 验证结果
            assert mock_scheduler_output.scheduled_new_reqs[0].additional_information == mock_payload
            assert "req_1" not in mock_connector.finished_requests    

    def test_get_chunk_new_requests_finished(self, mock_connector, mock_scheduler_output):
        """测试 get_chunk 处理已完成的新请求"""
        # 设置测试数据
        mock_connector.get_requests = {"req_1": 0}
        
        # 【必须添加】清空缓存请求列表，避免KeyError
        mock_scheduler_output.scheduled_cached_reqs.req_ids = []
        
        # 模拟 get_through_connector 返回完成的数据
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": True
        }
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.get_through_connector', 
                return_value=mock_payload):
            # 调用 get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # 验证结果
            assert mock_scheduler_output.scheduled_new_reqs[0].additional_information == mock_payload
            assert "req_1" in mock_connector.finished_requests
    
    def test_get_chunk_cached_requests(self, mock_connector, mock_scheduler_output):
        """测试 get_chunk 处理缓存请求"""
        # 设置测试数据
        mock_connector.get_requests = {"req_2": 0, "req_3": 0}
        # 【关键修复】清空新请求列表，避免KeyError
        mock_scheduler_output.scheduled_new_reqs = []
        # 模拟 get_through_connector 返回数据
        mock_payload = {
            "code_predictor_codes": [4, 5, 6],
            "finished": False
        }
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.get_through_connector', 
                  return_value=mock_payload):
            # 调用 get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # 验证结果
            cached_reqs = mock_scheduler_output.scheduled_cached_reqs
            assert cached_reqs.additional_informations["req_2"] == mock_payload
            assert cached_reqs.additional_informations["req_3"] == mock_payload
    
    def test_get_through_connector_success(self, mock_connector):
        """测试 get_through_connector 成功获取数据"""
        # 模拟 connector.get 返回有效数据
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "thinker_input_ids": [4, 5, 6]
        }
        mock_connector.get.return_value = (mock_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.request_prompt_token_ids = {}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # 调用 get_through_connector
            result = get_through_connector(
                mock_connector, 
                target_stage_id=0, 
                stage_id=1, 
                req_id="req_1", 
                connector_get_key="req_1_0_0"
            )
            
            # 验证结果
            assert result == mock_payload
            assert mock_connector.get_requests["req_1"] == 1
            assert mock_connector.request_prompt_token_ids["req_1"] == [4, 5, 6]
    
    def test_get_through_connector_stage_2_validation(self, mock_connector):
        """测试 stage_id=2 时的验证逻辑"""
        # 模拟 connector.get 返回有效数据（token_count 是16的倍数）
        mock_payload = {
            "code_predictor_codes": list(range(32)),  # 32是16的倍数
        }
        mock_connector.get.return_value = (mock_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.stage_id = 2
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # 调用 get_through_connector
            result = get_through_connector(
                mock_connector, 
                target_stage_id=1, 
                stage_id=2, 
                req_id="req_1", 
                connector_get_key="req_1_1_0"
            )
            
            # 验证结果
            assert result == mock_payload
    
    def test_get_through_connector_stage_2_invalid(self, mock_connector):
        """测试 stage_id=2 时验证失败的情况"""
        # 模拟 connector.get 始终返回无效数据（token_count 不是16的倍数）
        invalid_payload = {
            "code_predictor_codes": list(range(17)),  # 17不是16的倍数
        }
        
        # 关键：模拟connector.get始终返回相同无效数据
        mock_connector.get.return_value = (invalid_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.stage_id = 2
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            with patch('vllm_omni.distributed.omni_connectors.adapter.time.sleep'):
                # 调用 get_through_connector
                result = get_through_connector(
                    mock_connector, 
                    target_stage_id=1, 
                    stage_id=2, 
                    req_id="req_1", 
                    connector_get_key="req_1_1_0"
                )
                
                # 【正确断言】300次循环后，返回最后获取到的数据
                assert result == invalid_payload  # 不是None！
                
                # 验证：由于每次都是continue，get_requests计数不应该增加
                assert mock_connector.get_requests["req_1"] == 0
                
                # 验证：connector.get应该被调用了300次（max_wait）
                assert mock_connector.get.call_count == 300
        
    def test_validate_talker_output_valid(self):
        """测试 validate_talker_output 验证有效数据"""
        # 有效数据：token_count 是16的倍数
        payload_data = {
            "code_predictor_codes": list(range(32))
        }
        
        result = validate_talker_output(payload_data)
        assert result is True
    
    def test_validate_talker_output_invalid(self):
        """测试 validate_talker_output 验证无效数据"""
        # 无效数据：token_count 不是16的倍数
        payload_data = {
            "code_predictor_codes": list(range(15))
        }
        
        result = validate_talker_output(payload_data)
        assert result is False
    
    def test_validate_talker_output_empty(self):
        """测试 validate_talker_output 验证空数据"""
        # 空数据
        payload_data = {
            "code_predictor_codes": []
        }
        
        result = validate_talker_output(payload_data)
        assert result is False
    
    def test_get_chunk_for_generation(self, mock_connector, mock_request):
        """测试 get_chunk_for_generation"""
        # 设置测试数据
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.finished_requests = set()
        
        # 模拟 get_through_connector 返回数据
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": False
        }
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.get_through_connector', 
                  return_value=mock_payload):
            # 调用 get_chunk_for_generation
            result = get_chunk_for_generation(mock_connector, mock_request)
            
            # 验证结果
            assert mock_request.prompt_token_ids == [1, 2, 3]
            assert mock_request.status is None
            assert "req_1" not in mock_connector.finished_requests
    
    def test_get_chunk_for_generation_finished(self, mock_connector, mock_request):
        """测试 get_chunk_for_generation 处理已完成请求"""
        # 设置测试数据
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.finished_requests = set()
        
        # 模拟 get_through_connector 返回完成数据
        mock_payload = {
            "code_predictor_codes": [4, 5, 6],
            "finished": True
        }
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.get_through_connector', 
                  return_value=mock_payload):
            # 调用 get_chunk_for_generation
            result = get_chunk_for_generation(mock_connector, mock_request)
            
            # 验证结果
            assert mock_request.prompt_token_ids == [4, 5, 6]
            assert mock_request.status == RequestStatus.FINISHED_STOPPED
            assert "req_1" in mock_connector.finished_requests
    
    def test_get_chunk_for_generation_already_finished(self, mock_connector, mock_request):
        """测试 get_chunk_for_generation 处理已完成的请求（直接返回）"""
        # 设置请求已完成
        mock_connector.finished_requests = {"req_1"}
        
        # 调用 get_chunk_for_generation
        result = get_chunk_for_generation(mock_connector, mock_request)
        
        # 验证结果（应该直接返回，不执行后续逻辑）
        assert result is None
    
    def test_put_chunk_success(self, mock_connector, mock_request):
        """测试 put_chunk 成功发送数据"""
        # 设置测试数据
        mock_connector.put_requests = {"req_1": 0}
        mock_connector.request_prompt_token_ids = {}
        mock_request.prompt_token_ids = [1, 2, 3]
        
        # 模拟自定义处理函数
        def mock_process_func(pooling_output, request):
            return {
                "data": "test_data",
                "request_id": request.request_id
            }
        
        # 模拟 pooling_output
        mock_pooling_output = {"output": "test_output"}
        
        # 模拟 connector.put 成功
        mock_connector.put.return_value = (True, 100, {"metadata": "test"})
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # 调用 put_chunk
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            # 验证结果
            mock_connector.put.assert_called_once()
            assert mock_connector.put_requests["req_1"] == 1
            assert mock_connector.request_prompt_token_ids["req_1"] == [1, 2, 3]
    
    def test_put_chunk_no_payload(self, mock_connector, mock_request):
        """测试 put_chunk 无有效负载数据的情况"""
        # 设置测试数据
        mock_connector.put_requests = {"req_1": 0}
        
        # 模拟自定义处理函数返回None
        def mock_process_func(pooling_output, request):
            return None
        
        # 模拟 pooling_output
        mock_pooling_output = {"output": "test_output"}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # 调用 put_chunk
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            # 验证结果（应该记录警告但不发送数据）
            mock_connector.put.assert_not_called()
    
    def test_put_chunk_custom_func_exception(self, mock_connector, mock_request):
        """测试 put_chunk 自定义函数抛出异常的情况"""
        # 设置测试数据
        mock_connector.put_requests = {"req_1": 0}
        
        # 模拟自定义处理函数抛出异常
        def mock_process_func(pooling_output, request):
            raise ValueError("Test error")
        
        # 模拟 pooling_output
        mock_pooling_output = {"output": "test_output"}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # 调用 put_chunk
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            # 验证结果（应该记录错误但不发送数据）
            mock_connector.put.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])