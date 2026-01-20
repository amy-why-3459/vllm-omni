# test_entrypoints_omni.py
import pytest
from unittest.mock import Mock
from vllm_omni.entrypoints.omni import Omni

class TestIsAsyncChunkEnable:
    def test_async_chunk_true(self):
        """Normal case: async_chunk=True"""
        engine_args = Mock(async_chunk=True)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is True

    def test_async_chunk_false(self):
        """Normal case: async_chunk=False"""
        engine_args = Mock(async_chunk=False)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False

    def test_no_engine_args(self):
        """engine_args attribute does not exist"""
        stage_arg = Mock()
        del stage_arg.engine_args
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False

    def test_empty_stage_args(self):
        """Empty input list - should return False instead of crashing"""
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([])
        assert result is False

    def test_async_chunk_none(self):
        """async_chunk=None should return False"""
        engine_args = Mock(async_chunk=None)
        stage_arg = Mock(engine_args=engine_args)
        engine = Omni.__new__(Omni)
        result = engine._is_async_chunk_enable([stage_arg])
        assert result is False
