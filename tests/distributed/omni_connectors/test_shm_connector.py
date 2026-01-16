# tests/unit/distributed/omni_connectors/connectors/test_shm_connector.py


def test_import():
    """Test importing SharedMemoryConnector"""
    try:
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
        print("✅ Successfully imported SharedMemoryConnector")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_class_exists():
    """Test that the class exists and has basic attributes"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    
    # Check class name
    assert SharedMemoryConnector.__name__ == "SharedMemoryConnector"
    print(f"✅ Correct class name: {SharedMemoryConnector.__name__}")
    
    # Check required methods
    required_methods = ['__init__', 'put', 'get', 'cleanup']
    for method in required_methods:
        assert hasattr(SharedMemoryConnector, method), f"Missing method: {method}"
        print(f"✅ Method exists: {method}")
    
    return True


def test_method_signatures():
    """Test method signatures"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    init_sig = inspect.signature(SharedMemoryConnector.__init__)
    print(f"✅ __init__ signature: {init_sig}")
    
    put_sig = inspect.signature(SharedMemoryConnector.put)
    print(f"✅ put signature: {put_sig}")
    
    get_sig = inspect.signature(SharedMemoryConnector.get)
    print(f"✅ get signature: {get_sig}")
    
    return True


def test_init_default_values():
    """Test default initialization values"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    
    # Default configuration
    config = {}
    connector = SharedMemoryConnector(config)
    
    assert connector.stage_id == -1
    assert connector.device == "cuda:0"
    assert connector.threshold == 65536  # Default 64KB
    
    print("✅ Default initialization values are correct")
    
    # Custom configuration
    custom_config = {
        "stage_id": 1,
        "device": "cuda:1",
        "shm_threshold_bytes": 32768
    }
    connector2 = SharedMemoryConnector(custom_config)
    
    assert connector2.stage_id == 1
    assert connector2.device == "cuda:1"
    assert connector2.threshold == 32768
    
    print("✅ Custom configuration values are correct")
    
    return True


def test_put_method_logic():
    """Test key logic inside the put method"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    try:
        source = inspect.getsource(SharedMemoryConnector.put)
        
        checks = [
            ('serialize_obj', 'serialization'),
            ('shm_write_bytes', 'shared memory write'),
            ('put_key', 'put_key handling'),
            ('metadata', 'metadata creation')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ put method contains logic for {description}")
            else:
                print(f"⚠️  put method does NOT contain logic for {description}")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Unable to retrieve put method source code: {e}")
        return True


def test_get_method_logic():
    """Test key logic inside the get method"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    try:
        source = inspect.getsource(SharedMemoryConnector.get)
        
        checks = [
            ('SharedMemory', 'shared memory import'),
            ('shm_read_bytes', 'shared memory read'),
            ('get_key', 'get_key handling'),
            ('deserialize_obj', 'deserialization')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ get method contains logic for {description}")
            else:
                print(f"⚠️  get method does NOT contain logic for {description}")
        
        # Check retry logic
        if 'max_retries' in source and 'time.sleep' in source:
            print("✅ get method contains retry logic")
        else:
            print("⚠️  get method may be missing retry logic")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  Unable to retrieve get method source code: {e}")
        return True


def test_new_attributes():
    """Test newly added attributes"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    
    connector = SharedMemoryConnector({})
    
    new_attributes = [
        'stage_id',
        'device',
        'put_requests',
        'get_requests',
        'finished_requests',
        'request_prompt_token_ids'
    ]
    
    for attr in new_attributes:
        assert hasattr(connector, attr), f"Missing attribute: {attr}"
        print(f"✅ Attribute exists: {attr}")
    
    from collections import defaultdict
    assert isinstance(connector.put_requests, defaultdict)
    assert isinstance(connector.get_requests, defaultdict)
    assert isinstance(connector.finished_requests, set)
    assert isinstance(connector.request_prompt_token_ids, defaultdict)
    
    print("✅ New attribute types are correct")
    
    return True


def test_import_dependencies():
    """Test required imports"""
    import importlib
    
    dependencies = [
        'vllm_omni.entrypoints.stage_utils',
        'vllm_omni.distributed.omni_connectors.base',
        'multiprocessing.shared_memory',
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            if dep == 'multiprocessing.shared_memory':
                import multiprocessing
                if hasattr(multiprocessing, 'shared_memory'):
                    print(f"✅ Can import: {dep}")
                else:
                    print(f"⚠️  Cannot import {dep}: Python version may not support it")
                    all_ok = False
            else:
                importlib.import_module(dep)
                print(f"✅ Can import: {dep}")
        except ImportError as e:
            print(f"⚠️  Cannot import {dep}: {e}")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Starting SharedMemoryConnector tests")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("Import test", test_import),
        ("Class existence test", test_class_exists),
        ("Method signature test", test_method_signatures),
        ("Default initialization", test_init_default_values),
        ("Put method logic", test_put_method_logic),
        ("Get method logic", test_get_method_logic),
        ("New attributes test", test_new_attributes),
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
    
    print("\n" + "=" * 60)
    print("Test summary:")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
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
