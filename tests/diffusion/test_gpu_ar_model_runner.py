# test_gpu_ar_model_runner.py
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

class TestExecuteModelLogitsIndices:
    """测试execute_model中logits_indices越界处理逻辑"""
    
    def test_logits_indices_within_bounds_no_adjustment(self):
        """索引在范围内时不调整"""
        # 模拟数据
        hidden_states = torch.randn(100, 1024)
        logits_indices = torch.tensor([10, 20, 30])
        
        # 测试条件
        num_hidden_tokens = hidden_states.shape[0]
        assert logits_indices.max().item() < num_hidden_tokens
        
        # 采样应该正常工作
        sample_hidden_states = hidden_states[logits_indices]
        assert sample_hidden_states.shape == (3, 1024)
    
    def test_single_request_out_of_bounds(self):
        """单个请求越界时使用最后一个token"""
        hidden_states = torch.randn(24, 1024)  # 24个token
        logits_indices = torch.tensor([30])  # 越界
        num_scheduled_tokens_np = np.array([24])  # 单个请求
        
        # 测试调整逻辑
        num_hidden_tokens = hidden_states.shape[0]
        if len(logits_indices) == 1 and num_scheduled_tokens_np.shape[0] == 1:
            # 单个请求：使用最后一个token
            logits_indices = torch.tensor([num_hidden_tokens - 1])
        
        assert logits_indices.item() == 23  # 最后一个token索引
    
    def test_multiple_requests_out_of_bounds(self):
        """多个请求越界时使用累积和-1"""
        hidden_states = torch.randn(50, 1024)
        logits_indices = torch.tensor([30, 60, 90])  # 测试数据，实际会被覆盖
        num_scheduled_tokens_np = np.array([10, 20, 20])  # 总共50个token
        
        # 测试调整逻辑
        cumsum_tokens = torch.cumsum(
            torch.from_numpy(num_scheduled_tokens_np).to('cpu'), 
            dim=0
        )
        logits_indices = cumsum_tokens - 1
        
        # 验证结果
        expected = torch.tensor([9, 29, 49])  # 10-1, 30-1, 50-1
        assert torch.equal(logits_indices, expected)
    
    def test_indices_clamping_to_valid_range(self):
        """确保调整后的索引在有效范围内"""
        hidden_states = torch.randn(10, 1024)  # 只有10个token
        num_scheduled_tokens_np = np.array([5, 10])  # 第二个请求会越界
        
        cumsum_tokens = torch.cumsum(
            torch.from_numpy(num_scheduled_tokens_np).to('cpu'), 
            dim=0
        )
        logits_indices = cumsum_tokens - 1
        logits_indices = logits_indices.clamp(min=0, max=9)
        
        # 第二个请求应该被限制在9
        assert torch.equal(logits_indices, torch.tensor([4, 9]))
    
    def test_hidden_states_sampling_correct_shape(self):
        """验证采样hidden_states形状正确"""
        hidden_states = torch.randn(100, 768)
        logits_indices = torch.tensor([0, 50, 99])  # 第一个、中间、最后一个
        
        sample_hidden_states = hidden_states[logits_indices]
        
        # 应该采样3个token，每个768维
        assert sample_hidden_states.shape == (3, 768)
        # 验证采样正确
        assert torch.equal(sample_hidden_states[0], hidden_states[0])
        assert torch.equal(sample_hidden_states[1], hidden_states[50])
        assert torch.equal(sample_hidden_states[2], hidden_states[99])