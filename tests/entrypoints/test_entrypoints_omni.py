# test_entrypoints_omni.py
import pytest
from unittest.mock import Mock
from vllm_omni.entrypoints.omni import Omni

class TestIsAsyncChunkEnable:
    def test_async_chunk_true(self):
        """正常情况：async_chunk=True"""
        engine_args = Mock(async_chunk=True)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is True

    def test_async_chunk_false(self):
        """正常情况：async_chunk=False"""
        engine_args = Mock(async_chunk=False)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False

    def test_no_engine_args(self):
        """没有engine_args属性"""
        stage_arg = Mock()
        del stage_arg.engine_args
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False

    def test_empty_stage_args(self):
        """空列表输入 - 应该返回False而不是崩溃"""
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([])
        assert result is False

    def test_async_chunk_none(self):
        """async_chunk=None 应该返回 False"""
        engine_args = Mock(async_chunk=None)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False