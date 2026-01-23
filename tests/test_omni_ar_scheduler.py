# tests/unit/core/sched/test_omni_ar_scheduler.py

import inspect


def test_import():
    """OmniARScheduler can be imported."""
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
    assert OmniARScheduler is not None


def test_class_and_methods_exist():
    """Public class and required methods exist."""
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    assert OmniARScheduler.__name__ == "OmniARScheduler"

    # Only check public API, not implementation details
    for method in ("schedule", "update_from_output"):
        assert hasattr(OmniARScheduler, method), f"Missing method: {method}"


def test_method_signatures_are_callable():
    """Methods have stable callable signatures."""
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    # Do NOT assert exact parameters, just that they are functions
    assert callable(OmniARScheduler.schedule)
    assert callable(OmniARScheduler.update_from_output)


def test_dependencies_importable():
    """Key omni dependencies can be imported."""
    import importlib

    deps = [
        "vllm_omni.distributed.omni_connectors.adapter",
        "vllm_omni.distributed.omni_connectors.factory",
    ]

    for dep in deps:
        importlib.import_module(dep)
