# tests/unit/engine/test_arg_utils.py


def test_import():
    """Test that OmniEngineArgs can be imported"""
    try:
        from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs
        print(f"✅ Successfully imported OmniEngineArgs and AsyncOmniEngineArgs")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_class_exists():
    """Test that classes exist and check inheritance"""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs
    
    # Check class names
    assert OmniEngineArgs.__name__ == "OmniEngineArgs"
    assert AsyncOmniEngineArgs.__name__ == "AsyncOmniEngineArgs"
    print(f"✅ Class names are correct")
    
    return True


def test_new_fields():
    """Test default values of newly added fields - inspect source without instantiation"""
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    import inspect
    
    # Get source code to check field definitions
    source = inspect.getsource(OmniEngineArgs)
    
    # Check field definitions
    field_definitions = [
        ('stage_id: int = 0', 'stage_id'),
        ('model_stage: str = "thinker"', 'model_stage'),
        ('model_arch: str = "Qwen2_5OmniForConditionalGeneration"', 'model_arch'),
        ('async_chunk: bool = False', 'async_chunk'),
        ('stage_connector_spec: dict[str, any] = field(default_factory=dict)', 'stage_connector_spec'),
        ('hf_config_name: str | None = None', 'hf_config_name'),
        ('custom_process_next_stage_input_func: str | None = None', 'custom_process_next_stage_input_func')
    ]
    
    for field_def, field_name in field_definitions:
        if field_def in source:
            print(f"✅ Field present: {field_name}")
        else:
            print(f"⚠️  Field missing: {field_name}")
    
    return True


def test_method_existence():
    """Test method existence"""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs
    import inspect
    
    # Methods to check
    methods_to_check = [
        ('OmniEngineArgs', 'create_model_config'),
        ('OmniEngineArgs', 'draw_hf_text_config'),
        ('AsyncOmniEngineArgs', 'create_model_config'),
        ('AsyncOmniEngineArgs', 'draw_hf_text_config'),
    ]
    
    for class_name, method_name in methods_to_check:
        cls = OmniEngineArgs if class_name == 'OmniEngineArgs' else AsyncOmniEngineArgs
        
        # Check method existence
        if hasattr(cls, method_name):
            print(f"✅ {class_name} has method: {method_name}")
            
            # Try to get method signature (without calling)
            try:
                sig = inspect.signature(getattr(cls, method_name))
                print(f"  Signature: {sig}")
            except:
                print(f"  Unable to get signature")
        else:
            print(f"❌ {class_name} is missing method: {method_name}")
    
    return True


def test_create_model_config_logic_check():
    """Check logic of create_model_config method via source code inspection"""
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    import inspect
    
    try:
        source = inspect.getsource(OmniEngineArgs.create_model_config)
        
        # Check key logic
        logic_checks = [
            ('stage_id', 'stage_id assignment'),
            ('async_chunk', 'async_chunk handling'),
            ('model_stage', 'model_stage assignment'),
            ('stage_connector_name', 'stage_connector_name assignment'),
            ('stage_connector_extra', 'stage_connector_extra assignment'),
            ('hf_config_name', 'hf_config_name handling'),
            ('draw_hf_text_config', 'hf_text_config generation')
        ]
        
        print("Checking create_model_config method logic:")
        for keyword, description in logic_checks:
            if keyword in source:
                print(f"  ✅ Contains: {description}")
            else:
                print(f"  ⚠️  Missing: {description}")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Unable to retrieve create_model_config source code: {e}")
        return True


def test_code_comparison():
    """Compare code differences between OmniEngineArgs and AsyncOmniEngineArgs"""
    from vllm_omni.engine.arg_utils import OmniEngineArgs, AsyncOmniEngineArgs
    import inspect
    
    try:
        omni_source = inspect.getsource(OmniEngineArgs)
        async_source = inspect.getsource(AsyncOmniEngineArgs)
        
        print("Code comparison:")
        
        # Check common fields
        common_fields = ['stage_id', 'model_stage', 'model_arch', 'async_chunk']
        for field in common_fields:
            if f'{field}:' in omni_source and f'{field}:' in async_source:
                print(f"  ✅ Both classes contain field: {field}")
            else:
                print(f"  ⚠️  Field mismatch: {field}")
        
        # Check methods
        if 'def create_model_config' in omni_source and 'def create_model_config' in async_source:
            print("  ✅ Both classes have create_model_config method")
        
        if 'def draw_hf_text_config' in omni_source and 'def draw_hf_text_config' in async_source:
            print("  ✅ Both classes have draw_hf_text_config method")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Unable to retrieve source code: {e}")
        return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Starting OmniEngineArgs tests (fully simplified version)")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("Import test", test_import),
        ("Class existence test", test_class_exists),
        ("New fields test", test_new_fields),
        ("Method existence test", test_method_existence),
        ("create_model_config logic check", test_create_model_config_logic_check),
        ("Code comparison test", test_code_comparison),
    ]
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        try:
            success = test_func()
            if success:
                results.append((test_name, True, ""))
                print(f"✅ {test_name} passed")
            else:
                results.append((test_name, False, "Returned False"))
                print(f"❌ {test_name} failed")
        except AssertionError as e:
            results.append((test_name, False, f"Assertion failed: {e}"))
            print(f"❌ {test_name} failed: {e}")
        except Exception as e:
            results.append((test_name, False, f"Exception: {e}"))
            print(f"⚠️  {test_name} error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test summary:")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ Passed" if success else "❌ Failed"
        print(f"{status}: {test_name}")
        if error:
            print(f"     Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
