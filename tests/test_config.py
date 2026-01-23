# tests/unit/config/test_model_config.py 

import pytest

def test_default_values():
    """basic default value test"""
    from vllm_omni.config.model import OmniModelConfig
    
    # Create a configuration instance with a local path
    config = OmniModelConfig(model="/path/to/model")

    # Only test the most crucial few fields
    assert config.stage_id == 0
    assert config.model_stage == "thinker"
    # engine_output_type default
    assert config.engine_output_type is None
    # new field omni_kv_config should default to None
    assert config.omni_kv_config is None
    # model_arch has a default
    assert isinstance(config.model_arch, str)


def test_custom_basic_values():
    """Test overriding basic values."""
    from vllm_omni.config.model import OmniModelConfig
    
    config = OmniModelConfig(
        model="/path/to/model",
        stage_id=1,
        model_stage="talker",
        engine_output_type="audio",
        omni_kv_config={"key": "value"},
    )
    
    assert config.stage_id == 1
    assert config.model_stage == "talker"
    assert config.engine_output_type == "audio"
    assert config.omni_kv_config == {"key": "value"}


def test_inheritance():
    """Confirm the inheritance relationship"""
    from vllm_omni.config.model import OmniModelConfig
    from vllm.config import ModelConfig
    
    config = OmniModelConfig(model="/path/to/model")
    assert isinstance(config, ModelConfig)
