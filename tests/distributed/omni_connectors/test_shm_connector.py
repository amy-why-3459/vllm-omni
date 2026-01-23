# tests/unit/distributed/omni_connectors/connectors/test_shm_connector.py


def test_shm_connector_import():
    """SharedMemoryConnector can be imported."""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
        SharedMemoryConnector,
    )

    assert SharedMemoryConnector is not None


def test_shm_connector_class_and_methods_exist():
    """SharedMemoryConnector exposes required public methods."""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
        SharedMemoryConnector,
    )

    assert SharedMemoryConnector.__name__ == "SharedMemoryConnector"

    for method in ("__init__", "put", "get", "cleanup"):
        assert hasattr(SharedMemoryConnector, method)
        assert callable(getattr(SharedMemoryConnector, method))


def test_shm_connector_minimal_init():
    """SharedMemoryConnector can be constructed with minimal config."""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
        SharedMemoryConnector,
    )

    connector = SharedMemoryConnector({})

    assert connector is not None
