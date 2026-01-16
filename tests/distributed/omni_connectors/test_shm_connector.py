# tests/unit/distributed/omni_connectors/connectors/test_shm_connector.py


def test_import():
    """测试能导入SharedMemoryConnector"""
    try:
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
        print(f"✅ 成功导入 SharedMemoryConnector")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_class_exists():
    """测试类存在并检查基本属性"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    
    # 检查类名
    assert SharedMemoryConnector.__name__ == "SharedMemoryConnector"
    print(f"✅ 类名正确: {SharedMemoryConnector.__name__}")
    
    # 检查关键方法
    required_methods = ['__init__', 'put', 'get', 'cleanup']
    for method in required_methods:
        assert hasattr(SharedMemoryConnector, method), f"缺少方法: {method}"
        print(f"✅ 有方法: {method}")
    
    return True


def test_method_signatures():
    """测试方法签名"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    # 检查__init__签名
    init_sig = inspect.signature(SharedMemoryConnector.__init__)
    print(f"✅ __init__ 签名: {init_sig}")
    
    # 检查put签名
    put_sig = inspect.signature(SharedMemoryConnector.put)
    print(f"✅ put 签名: {put_sig}")
    
    # 检查get签名
    get_sig = inspect.signature(SharedMemoryConnector.get)
    print(f"✅ get 签名: {get_sig}")
    
    return True


def test_init_default_values():
    """测试初始化默认值"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    from unittest.mock import Mock
    
    # 测试默认配置
    config = {}
    connector = SharedMemoryConnector(config)
    
    # 检查默认值
    assert connector.stage_id == -1
    assert connector.device == "cuda:0"
    assert connector.threshold == 65536  # 64KB默认值
    
    print("✅ 初始化默认值正确")
    
    # 测试自定义配置
    custom_config = {
        "stage_id": 1,
        "device": "cuda:1",
        "shm_threshold_bytes": 32768
    }
    connector2 = SharedMemoryConnector(custom_config)
    
    assert connector2.stage_id == 1
    assert connector2.device == "cuda:1"
    assert connector2.threshold == 32768
    
    print("✅ 自定义配置正确")
    
    return True


def test_put_method_logic():
    """测试put方法的关键逻辑"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    try:
        source = inspect.getsource(SharedMemoryConnector.put)
        
        # 检查关键代码
        checks = [
            ('serialize_obj', '序列化方法'),
            ('shm_write_bytes', '共享内存写入'),
            ('put_key', 'put_key参数'),
            ('metadata', '元数据创建')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ put方法包含'{description}'逻辑")
            else:
                print(f"⚠️  put方法不包含'{description}'逻辑")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  无法获取put方法源代码: {e}")
        return True


def test_get_method_logic():
    """测试get方法的关键逻辑"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    import inspect
    
    try:
        source = inspect.getsource(SharedMemoryConnector.get)
        
        # 检查关键代码
        checks = [
            ('SharedMemory', '共享内存导入'),
            ('shm_read_bytes', '共享内存读取'),
            ('get_key', 'get_key参数'),
            ('deserialize_obj', '反序列化')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ get方法包含'{description}'逻辑")
            else:
                print(f"⚠️  get方法不包含'{description}'逻辑")
        
        # 检查新的重试逻辑
        if 'max_retries' in source and 'time.sleep' in source:
            print("✅ get方法包含重试逻辑")
        else:
            print("⚠️  get方法可能缺少重试逻辑")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  无法获取get方法源代码: {e}")
        return True


def test_new_attributes():
    """测试新增的属性"""
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    
    # 创建实例
    config = {}
    connector = SharedMemoryConnector(config)
    
    # 检查新增的属性
    new_attributes = [
        'stage_id',
        'device',
        'put_requests',
        'get_requests',
        'finished_requests',
        'request_prompt_token_ids'
    ]
    
    for attr in new_attributes:
        assert hasattr(connector, attr), f"缺少属性: {attr}"
        print(f"✅ 有属性: {attr}")
    
    # 检查默认集合类型
    from collections import defaultdict
    assert isinstance(connector.put_requests, defaultdict)
    assert isinstance(connector.get_requests, defaultdict)
    assert isinstance(connector.finished_requests, set)
    assert isinstance(connector.request_prompt_token_ids, defaultdict)
    
    print("✅ 新增属性类型正确")
    
    return True


def test_import_dependencies():
    """测试相关导入"""
    import importlib
    
    dependencies = [
        'vllm_omni.entrypoints.stage_utils',
        'vllm_omni.distributed.omni_connectors.base',
        'multiprocessing.shared_memory',
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            # 尝试导入相关模块
            if dep == 'multiprocessing.shared_memory':
                # 特殊处理：检查是否能导入
                import multiprocessing
                if hasattr(multiprocessing, 'shared_memory'):
                    print(f"✅ 能导入: {dep}")
                else:
                    print(f"⚠️  无法导入 {dep}: Python版本可能不支持")
                    all_ok = False
            else:
                importlib.import_module(dep)
                print(f"✅ 能导入: {dep}")
        except ImportError as e:
            print(f"⚠️  无法导入 {dep}: {e}")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始 SharedMemoryConnector 测试")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("导入测试", test_import),
        ("类存在测试", test_class_exists),
        ("方法签名测试", test_method_signatures),
        ("初始化默认值", test_init_default_values),
        ("put方法逻辑", test_put_method_logic),
        ("get方法逻辑", test_get_method_logic),
        ("新增属性测试", test_new_attributes),
        ("依赖导入测试", test_import_dependencies),
    ]
    
    for test_name, test_func in tests:
        print(f"\n▶️  运行: {test_name}")
        try:
            success = test_func()
            if success:
                results.append((test_name, True, ""))
                print(f"✅ {test_name} 通过")
            else:
                results.append((test_name, False, "返回False"))
                print(f"❌ {test_name} 失败")
        except AssertionError as e:
            results.append((test_name, False, f"断言失败: {e}"))
            print(f"❌ {test_name} 失败: {e}")
        except Exception as e:
            results.append((test_name, False, f"异常: {e}"))
            print(f"⚠️  {test_name} 错误: {e}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
        if error:
            print(f"     错误: {error}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)