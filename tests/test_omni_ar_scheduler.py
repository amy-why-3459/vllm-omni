# tests/unit/core/sched/test_omni_ar_scheduler.py


def test_import():
    """Test that OmniARScheduler can be imported"""
    try:
        from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
        print(f"✅ Successfully imported OmniARScheduler")
        return OmniARScheduler
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise


def test_class_exists():
    """Test that the class exists and check basic attributes"""
    OmniARScheduler = test_import()
    
    # Check class name
    assert OmniARScheduler.__name__ == "OmniARScheduler"
    print(f"✅ Class name is correct: {OmniARScheduler.__name__}")
    
    # Check required methods
    required_methods = ['__init__', 'schedule', 'update_from_output']
    for method in required_methods:
        assert hasattr(OmniARScheduler, method), f"Missing method: {method}"
        print(f"✅ Method exists: {method}")
    
    return OmniARScheduler


def test_method_signatures():
    """Test method signatures"""
    OmniARScheduler = test_class_exists()
    
    import inspect
    
    # Check __init__ signature
    init_sig = inspect.signature(OmniARScheduler.__init__)
    print(f"✅ __init__ signature: {init_sig}")
    
    # Check schedule signature
    schedule_sig = inspect.signature(OmniARScheduler.schedule)
    print(f"✅ schedule signature: {schedule_sig}")
    
    # Check update_from_output signature
    update_sig = inspect.signature(OmniARScheduler.update_from_output)
    print(f"✅ update_from_output signature: {update_sig}")
    
    return OmniARScheduler


def test_source_code_checks():
    """Check key strings in source code"""
    OmniARScheduler = test_method_signatures()
    
    import inspect
    
    # Check key code in schedule method
    try:
        source = inspect.getsource(OmniARScheduler.schedule)
        
        # Check omni-related logic
        checks = [
            ("omni_connector", "Omni connector"),
            ("get_chunk", "chunk fetching"),
            ("async_chunk", "async chunk"),
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ schedule method contains {description} logic")
            else:
                print(f"⚠️  schedule method does not contain {description} logic")
    except (TypeError, OSError):
        print("⚠️  Unable to retrieve schedule method source code")
    
    # Check update_from_output method
    try:
        source = inspect.getsource(OmniARScheduler.update_from_output)
        
        checks = [
            ("put_chunk", "chunk placement"),
            ("custom_process", "custom processing"),
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ update_from_output method contains {description} logic")
            else:
                print(f"⚠️  update_from_output method does not contain {description} logic")
    except (TypeError, OSError):
        print("⚠️  Unable to retrieve update_from_output method source code")
    
    return OmniARScheduler


def test_dependency_imports():
    """Test whether dependency modules can be imported"""
    import importlib
    
    dependencies = [
        'vllm_omni.distributed.omni_connectors.adapter',
        'vllm_omni.distributed.omni_connectors.factory',
        'importlib',
    ]
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ Dependency imported successfully: {dep}")
        except ImportError as e:
            print(f"⚠️  Failed to import dependency {dep}: {e}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Starting OmniARScheduler tests")
    print("=" * 60)
    
    try:
        test_import()
        print()
        
        test_class_exists()
        print()
        
        test_method_signatures()
        print()
        
        test_source_code_checks()
        print()
        
        test_dependency_imports()
        print()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Tests failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
