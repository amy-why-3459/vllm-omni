# tests/unit/core/sched/test_omni_generation_scheduler.py


def test_import():
    """OmniGenerationScheduler can be imported."""
    from vllm_omni.core.sched.omni_generation_scheduler import (
        OmniGenerationScheduler,
    )
    assert OmniGenerationScheduler is not None


def test_class_and_methods_exist():
    """Public class and required methods exist."""
    from vllm_omni.core.sched.omni_generation_scheduler import (
        OmniGenerationScheduler,
    )

    assert OmniGenerationScheduler.__name__ == "OmniGenerationScheduler"

    # Only check public API
    for method in ("schedule", "update_from_output"):
        assert hasattr(OmniGenerationScheduler, method), (
            f"Missing method: {method}"
        )


def test_methods_are_callable():
    """Scheduler methods are callable."""
    from vllm_omni.core.sched.omni_generation_scheduler import (
        OmniGenerationScheduler,
    )

    assert callable(OmniGenerationScheduler.schedule)
    assert callable(OmniGenerationScheduler.update_from_output)


def test_dependencies_importable():
    """Key omni dependencies can be imported."""
    import importlib

    deps = [
        "vllm_omni.distributed.omni_connectors.adapter",
        "vllm_omni.distributed.omni_connectors.factory",
        "vllm_omni.core.sched.output",
    ]

    for dep in deps:
        importlib.import_module(dep)
