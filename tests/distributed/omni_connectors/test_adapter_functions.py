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
    """Tests for OmniConnector adapter functions"""
    
    @pytest.fixture
    def mock_connector(self):
        """Create a mocked connector"""
        connector = Mock()
        connector.stage_id = 1
        connector.get_requests = {}
        connector.finished_requests = set()
        connector.put_requests = {}
        connector.request_prompt_token_ids = {}
        return connector
    
    @pytest.fixture
    def mock_scheduler_output(self):
        """Create a mocked scheduler_output"""
        output = Mock()
        
        # Mock new requests
        new_req = Mock()
        new_req.req_id = "req_1"
        new_req.additional_information = None
        output.scheduled_new_reqs = [new_req]
        
        # Mock cached requests
        cached_reqs = Mock()
        cached_reqs.req_ids = ["req_2", "req_3"]
        cached_reqs.additional_informations = {}
        output.scheduled_cached_reqs = cached_reqs
        
        return output

    @pytest.fixture
    def mock_request(self):
        """Create a mocked request"""
        request = Mock()
        request.request_id = "req_1"
        request.prompt_token_ids = []
        request.status = None
        return request
    
    def test_get_chunk_for_stage_0(self, mock_connector, mock_scheduler_output):
        """Test that get_chunk returns immediately when stage_id=0"""
        # Set stage_id=0
        mock_connector.stage_id = 0
        
        # Call get_chunk
        result = get_chunk(mock_connector, mock_scheduler_output)
        
        # stage_id=0 should return directly without further processing
        assert result is None
    
    def test_get_chunk_new_requests(self, mock_connector, mock_scheduler_output):
        """Test get_chunk handling new requests"""
        # Set test data
        mock_connector.get_requests = {"req_1": 0}
        
        # [Critical fix] Ensure cached request list is empty to avoid interference
        mock_scheduler_output.scheduled_cached_reqs.req_ids = []
        
        # Mock get_through_connector returning valid data
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": False
        }

        with patch(
            'vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
            return_value=mock_payload
        ):
            # Call get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # Verify results
            assert (
                mock_scheduler_output.scheduled_new_reqs[0].additional_information
                == mock_payload
            )
            assert "req_1" not in mock_connector.finished_requests    

    def test_get_chunk_new_requests_finished(self, mock_connector, mock_scheduler_output):
        """Test get_chunk handling finished new requests"""
        # Set test data
        mock_connector.get_requests = {"req_1": 0}
        
        # [Required] Clear cached requests to avoid KeyError
        mock_scheduler_output.scheduled_cached_reqs.req_ids = []
        
        # Mock get_through_connector returning finished data
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": True
        }
        
        with patch(
            'vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
            return_value=mock_payload
        ):
            # Call get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # Verify results
            assert (
                mock_scheduler_output.scheduled_new_reqs[0].additional_information
                == mock_payload
            )
            assert "req_1" in mock_connector.finished_requests
    
    def test_get_chunk_cached_requests(self, mock_connector, mock_scheduler_output):
        """Test get_chunk handling cached requests"""
        # Set test data
        mock_connector.get_requests = {"req_2": 0, "req_3": 0}
        
        # [Critical fix] Clear new request list to avoid KeyError
        mock_scheduler_output.scheduled_new_reqs = []
        
        # Mock get_through_connector returning data
        mock_payload = {
            "code_predictor_codes": [4, 5, 6],
            "finished": False
        }
        
        with patch(
            'vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
            return_value=mock_payload
        ):
            # Call get_chunk
            result = get_chunk(mock_connector, mock_scheduler_output)
            
            # Verify results
            cached_reqs = mock_scheduler_output.scheduled_cached_reqs
            assert cached_reqs.additional_informations["req_2"] == mock_payload
            assert cached_reqs.additional_informations["req_3"] == mock_payload
    
    def test_get_through_connector_success(self, mock_connector):
        """Test successful data retrieval via get_through_connector"""
        # Mock connector.get returning valid data
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "thinker_input_ids": [4, 5, 6]
        }
        mock_connector.get.return_value = (mock_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.request_prompt_token_ids = {}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # Call get_through_connector
            result = get_through_connector(
                mock_connector,
                target_stage_id=0,
                stage_id=1,
                req_id="req_1",
                connector_get_key="req_1_0_0"
            )
            
            # Verify results
            assert result == mock_payload
            assert mock_connector.get_requests["req_1"] == 1
            assert mock_connector.request_prompt_token_ids["req_1"] == [4, 5, 6]
    
    def test_get_through_connector_stage_2_validation(self, mock_connector):
        """Test validation logic when stage_id=2"""
        # Mock connector.get returning valid data (token_count is a multiple of 16)
        mock_payload = {
            "code_predictor_codes": list(range(32))
        }
        mock_connector.get.return_value = (mock_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.stage_id = 2
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            # Call get_through_connector
            result = get_through_connector(
                mock_connector,
                target_stage_id=1,
                stage_id=2,
                req_id="req_1",
                connector_get_key="req_1_1_0"
            )
            
            # Verify result
            assert result == mock_payload
    
    def test_get_through_connector_stage_2_invalid(self, mock_connector):
        """Test validation failure when stage_id=2"""
        # Mock connector.get always returning invalid data
        invalid_payload = {
            "code_predictor_codes": list(range(17))
        }
        
        mock_connector.get.return_value = (invalid_payload, 100)
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.stage_id = 2
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            with patch('vllm_omni.distributed.omni_connectors.adapter.time.sleep'):
                # Call get_through_connector
                result = get_through_connector(
                    mock_connector,
                    target_stage_id=1,
                    stage_id=2,
                    req_id="req_1",
                    connector_get_key="req_1_1_0"
                )
                
                # [Correct assertion] After 300 retries, return the last payload
                assert result == invalid_payload
                
                # get_requests counter should not increase
                assert mock_connector.get_requests["req_1"] == 0
                
                # connector.get should be called 300 times (max_wait)
                assert mock_connector.get.call_count == 300
        
    def test_validate_talker_output_valid(self):
        """Test validate_talker_output with valid data"""
        payload_data = {
            "code_predictor_codes": list(range(32))
        }
        
        result = validate_talker_output(payload_data)
        assert result is True
    
    def test_validate_talker_output_invalid(self):
        """Test validate_talker_output with invalid data"""
        payload_data = {
            "code_predictor_codes": list(range(15))
        }
        
        result = validate_talker_output(payload_data)
        assert result is False
    
    def test_validate_talker_output_empty(self):
        """Test validate_talker_output with empty data"""
        payload_data = {
            "code_predictor_codes": []
        }
        
        result = validate_talker_output(payload_data)
        assert result is False
    
    def test_get_chunk_for_generation(self, mock_connector, mock_request):
        """Test get_chunk_for_generation"""
        # Set test data
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.finished_requests = set()
        
        mock_payload = {
            "code_predictor_codes": [1, 2, 3],
            "finished": False
        }
        
        with patch(
            'vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
            return_value=mock_payload
        ):
            # Call get_chunk_for_generation
            result = get_chunk_for_generation(mock_connector, mock_request)
            
            # Verify results
            assert mock_request.prompt_token_ids == [1, 2, 3]
            assert mock_request.status is None
            assert "req_1" not in mock_connector.finished_requests
    
    def test_get_chunk_for_generation_finished(self, mock_connector, mock_request):
        """Test get_chunk_for_generation handling finished requests"""
        mock_connector.get_requests = {"req_1": 0}
        mock_connector.finished_requests = set()
        
        mock_payload = {
            "code_predictor_codes": [4, 5, 6],
            "finished": True
        }
        
        with patch(
            'vllm_omni.distributed.omni_connectors.adapter.get_through_connector',
            return_value=mock_payload
        ):
            result = get_chunk_for_generation(mock_connector, mock_request)
            
            assert mock_request.prompt_token_ids == [4, 5, 6]
            assert mock_request.status == RequestStatus.FINISHED_STOPPED
            assert "req_1" in mock_connector.finished_requests
    
    def test_get_chunk_for_generation_already_finished(self, mock_connector, mock_request):
        """Test get_chunk_for_generation with already finished requests"""
        mock_connector.finished_requests = {"req_1"}
        
        result = get_chunk_for_generation(mock_connector, mock_request)
        
        # Should return immediately without further processing
        assert result is None
    
    def test_put_chunk_success(self, mock_connector, mock_request):
        """Test successful data sending via put_chunk"""
        mock_connector.put_requests = {"req_1": 0}
        mock_connector.request_prompt_token_ids = {}
        mock_request.prompt_token_ids = [1, 2, 3]
        
        def mock_process_func(pooling_output, request):
            return {
                "data": "test_data",
                "request_id": request.request_id
            }
        
        mock_pooling_output = {"output": "test_output"}
        mock_connector.put.return_value = (True, 100, {"metadata": "test"})
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            mock_connector.put.assert_called_once()
            assert mock_connector.put_requests["req_1"] == 1
            assert mock_connector.request_prompt_token_ids["req_1"] == [1, 2, 3]
    
    def test_put_chunk_no_payload(self, mock_connector, mock_request):
        """Test put_chunk when no valid payload is produced"""
        mock_connector.put_requests = {"req_1": 0}
        
        def mock_process_func(pooling_output, request):
            return None
        
        mock_pooling_output = {"output": "test_output"}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            # Should log a warning but not send data
            mock_connector.put.assert_not_called()
    
    def test_put_chunk_custom_func_exception(self, mock_connector, mock_request):
        """Test put_chunk when the custom processing function raises an exception"""
        mock_connector.put_requests = {"req_1": 0}
        
        def mock_process_func(pooling_output, request):
            raise ValueError("Test error")
        
        mock_pooling_output = {"output": "test_output"}
        
        with patch('vllm_omni.distributed.omni_connectors.adapter.logger'):
            result = put_chunk(
                mock_connector,
                mock_pooling_output,
                mock_request,
                mock_process_func
            )
            
            # Should log an error but not send data
            mock_connector.put.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
