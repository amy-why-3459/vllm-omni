# tests/unit/distributed/omni_connectors/test_adapter.py


def test_adapter_imports():
    """Adapter public APIs can be imported."""
    from vllm_omni.distributed.omni_connectors.adapter import (
        get_chunk,
        get_through_connector,
        put_chunk,
    )

    assert get_chunk is not None
    assert get_through_connector is not None
    assert put_chunk is not None


def test_adapter_functions_callable():
    """Adapter functions are callable."""
    from vllm_omni.distributed.omni_connectors import adapter

    for fn_name in (
        "get_chunk",
        "get_through_connector",
        "put_chunk",
    ):
        assert hasattr(adapter, fn_name)
        assert callable(getattr(adapter, fn_name))


def test_adapter_dependency_imports():
    """Adapter dependencies can be imported."""
    import importlib

    deps = [
        "vllm_omni.distributed.omni_connectors.adapter",
        "vllm_omni.distributed.omni_connectors.factory",
    ]

    for dep in deps:
        importlib.import_module(dep)
