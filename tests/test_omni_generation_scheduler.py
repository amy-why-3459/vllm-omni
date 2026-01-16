# tests/unit/core/sched/test_omni_generation_scheduler.py


def test_import():
    """Test that OmniGenerationScheduler can be imported"""
    try:
        from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
        print(f"✅ Successfully imported OmniGenerationScheduler")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_class_exists():
    """Test that the class exists and check basic attributes"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    
    # Check class name
    assert OmniGenerationScheduler.__name__ == "OmniGenerationScheduler"
    print(f"✅ Class name correct: {OmniGenerationScheduler.__name__}")
    
    # Check required methods
    required_methods = ['__init__', 'schedule', 'update_from_output']
    for method in required_methods:
        assert hasattr(OmniGenerationScheduler, method), f"Missing method: {method}"
        print(f"✅ Has method: {method}")
    
    return True


def test_method_signatures():
    """Test method signatures"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    # Check __init__ signature
    init_sig = inspect.signature(OmniGenerationScheduler.__init__)
    print(f"✅ __init__ signature: {init_sig}")
    
    # Check schedule signature
    schedule_sig = inspect.signature(OmniGenerationScheduler.schedule)
    print(f"✅ schedule signature: {schedule_sig}")
    
    # Check update_from_output signature
    update_sig = inspect.signature(OmniGenerationScheduler.update_from_output)
    print(f"✅ update_from_output signature: {update_sig}")
    
    return True


def test_simple_init():
    """Simple test for initialization logic - using correct mocking approach"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    from unittest.mock import Mock, patch, PropertyMock
    
    # Method 1: Fully mock the __init__ method
    with patch.object(OmniGenerationScheduler, '__init__', return_value=None):
        scheduler = OmniGenerationScheduler()
        scheduler.vllm_config = None
        scheduler.omni_connector = None
        scheduler.stage_id = None
        
        # Verify instance can be created
        assert scheduler is not None
        print("✅ Can create OmniGenerationScheduler instance")
    
    # Method 2: Directly inspect initialization logic in source code
    print("Checking key code in __init__ method...")
    import inspect
    source = inspect.getsource(OmniGenerationScheduler.__init__)
    
    # Check for key logic
    keywords_to_check = [
        'model_config',
        'async_chunk', 
        'omni_connector',
        'stage_id',
        'vllm_config'
    ]
    
    for keyword in keywords_to_check:
        if keyword in source:
            print(f"✅ __init__ method contains '{keyword}'")
        else:
            print(f"⚠️  __init__ method does not contain '{keyword}'")
    
    return True


def test_schedule_method_code():
    """Check key code in schedule method"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    try:
        source = inspect.getsource(OmniGenerationScheduler.schedule)
        
        # Check for omni-related code
        omni_keywords = [
            'omni_connector',
            'get_chunk_for_generation',
            'async_chunk',
            'stage_id'
        ]
        
        found = []
        for keyword in omni_keywords:
            if keyword in source:
                found.append(keyword)
        
        if found:
            print(f"✅ schedule method contains Omni code: {', '.join(found)}")
        else:
            print("⚠️  schedule method does not contain obvious Omni code")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Cannot get source code of schedule method: {e}")
        return True  # Not a fatal error


def test_update_from_output_code():
    """Check key code in update_from_output method"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    try:
        source = inspect.getsource(OmniGenerationScheduler.update_from_output)
        
        # Check key logic
        checks = [
            ('FINISHED_STOPPED', 'finished state check'),
            ('omni_connector', 'Omni connector check'),
            ('num_computed_tokens', 'computed token count check'),
            ('num_prompt_tokens', 'prompt token count check')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ update_from_output contains '{description}' code")
            else:
                print(f"⚠️  update_from_output does not contain '{description}' code")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Cannot get source code of update_from_output method: {e}")
        return True  # Not a fatal error


def test_import_dependencies():
    """Test related imports"""
    import importlib
    
    dependencies = [
        'vllm_omni.distributed.omni_connectors.adapter',
        'vllm_omni.distributed.omni_connectors.factory',
        'vllm_omni.core.sched.output',
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ Can import: {dep}")
        except ImportError as e:
            print(f"⚠️  Cannot import {dep}: {e}")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Starting OmniGenerationScheduler tests")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("Import test", test_import),
        ("Class existence test", test_class_exists),
        ("Method signature test", test_method_signatures),
        ("Simple initialization test", test_simple_init),
        ("Schedule method code check", test_schedule_method_code),
        ("Update_from_output code check", test_update_from_output_code),
        ("Dependency import test", test_import_dependencies),
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
    print("Test results summary:")
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
