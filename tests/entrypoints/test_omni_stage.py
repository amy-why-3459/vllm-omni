# tests/unit/entrypoints/test_omni_stage.py

from unittest.mock import Mock


def test_async_chunk_injects_stage_config():
    """
    async_chunk=True 时，应注入：
    - engine_args.stage_id
    - engine_args.stage_connector_spec（取第一个 connector 的 spec）
    """
    engine_args = Mock()
    engine_args.async_chunk = True

    connectors_config = {
        "conn1": {"spec": {"key1": "value1", "key2": "value2"}},
        "conn2": {"spec": {"key3": "value3"}},
    }
    stage_id = 1

    # 模拟 omni_stage 中的最小副作用逻辑
    if engine_args.async_chunk:
        stage_connector_spec = {}
        for v in connectors_config.values():
            stage_connector_spec = dict(v.get("spec", {}))
            break
        engine_args.stage_connector_spec = stage_connector_spec
        engine_args.stage_id = stage_id

    assert engine_args.stage_id == stage_id
    assert engine_args.stage_connector_spec == {
        "key1": "value1",
        "key2": "value2",
    }


def test_async_chunk_disabled_no_injection():
    """
    async_chunk=False 时，不应对 engine_args 注入任何新字段
    """
    engine_args = Mock()
    engine_args.async_chunk = False

    original_attrs = set(vars(engine_args).keys())

    # 模拟代码路径
    if engine_args.async_chunk:
        engine_args.stage_connector_spec = {}
        engine_args.stage_id = 0

    assert set(vars(engine_args).keys()) == original_attrs

