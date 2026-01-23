# tests/unit/entrypoints/test_omni.py

from unittest.mock import Mock
from vllm_omni.entrypoints.omni import Omni


class TestIsAsyncChunkEnable:

    def test_async_chunk_enabled(self):
        """Any stage enables async_chunk -> True"""
        engine_args = Mock(async_chunk=True)
        stage_arg = Mock(engine_args=engine_args)

        engine = Omni.__new__(Omni)
        assert engine._is_async_chunk_enable([stage_arg]) is True

    def test_async_chunk_disabled(self):
        """async_chunk disabled -> False"""
        engine_args = Mock(async_chunk=False)
        stage_arg = Mock(engine_args=engine_args)

        engine = Omni.__new__(Omni)
        assert engine._is_async_chunk_enable([stage_arg]) is False

    def test_empty_stage_args(self):
        """Empty stage list should safely return False"""
        engine = Omni.__new__(Omni)
        assert engine._is_async_chunk_enable([]) is False
