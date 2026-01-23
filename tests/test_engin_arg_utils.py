# tests/unit/engine/test_arg_utils.py


def test_import():
    """OmniEngineArgs and AsyncOmniEngineArgs can be imported."""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs

    assert OmniEngineArgs is not None
    assert AsyncOmniEngineArgs is not None


def test_class_exists():
    """Classes exist with correct names."""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs

    assert OmniEngineArgs.__name__ == "OmniEngineArgs"
    assert AsyncOmniEngineArgs.__name__ == "AsyncOmniEngineArgs"


def test_minimal_instantiation():
    """Args classes can be instantiated with minimal arguments."""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs

    omni_args = OmniEngineArgs(model="dummy-model")
    async_args = AsyncOmniEngineArgs(model="dummy-model")

    assert omni_args is not None
    assert async_args is not None


def test_create_model_config_callable():
    """create_model_config can be called without crashing."""
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    args = OmniEngineArgs(model="dummy-model")

    # We only assert that the method can be called.
    # Internal fields are intentionally NOT checked.
    config = args.create_model_config()

    assert config is not None
