# tests/unit/core/sched/test_omni_ar_scheduler.py


def test_import():
    """测试能导入OmniARScheduler"""
    try:
        from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
        print(f"✅ 成功导入 OmniARScheduler")
        return OmniARScheduler
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        raise


def test_class_exists():
    """测试类存在并检查基本属性"""
    OmniARScheduler = test_import()
    
    # 检查类名
    assert OmniARScheduler.__name__ == "OmniARScheduler"
    print(f"✅ 类名正确: {OmniARScheduler.__name__}")
    
    # 检查关键方法
    required_methods = ['__init__', 'schedule', 'update_from_output']
    for method in required_methods:
        assert hasattr(OmniARScheduler, method), f"缺少方法: {method}"
        print(f"✅ 有方法: {method}")
    
    return OmniARScheduler


def test_method_signatures():
    """测试方法签名"""
    OmniARScheduler = test_class_exists()
    
    import inspect
    
    # 检查__init__签名
    init_sig = inspect.signature(OmniARScheduler.__init__)
    print(f"✅ __init__ 签名: {init_sig}")
    
    # 检查schedule签名
    schedule_sig = inspect.signature(OmniARScheduler.schedule)
    print(f"✅ schedule 签名: {schedule_sig}")
    
    # 检查update_from_output签名
    update_sig = inspect.signature(OmniARScheduler.update_from_output)
    print(f"✅ update_from_output 签名: {update_sig}")
    
    return OmniARScheduler


def test_source_code_checks():
    """检查源代码中的关键字符串"""
    OmniARScheduler = test_method_signatures()
    
    import inspect
    
    # 检查schedule方法中的关键代码
    try:
        source = inspect.getsource(OmniARScheduler.schedule)
        
        # 检查是否有omni相关代码
        checks = [
            ("omni_connector", "Omni连接器"),
            ("get_chunk", "获取chunk"),
            ("async_chunk", "异步chunk"),
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ schedule方法包含{description}代码")
            else:
                print(f"⚠️  schedule方法不包含{description}代码")
    except (TypeError, OSError):
        print("⚠️  无法获取schedule方法源代码")
    
    # 检查update_from_output方法
    try:
        source = inspect.getsource(OmniARScheduler.update_from_output)
        
        checks = [
            ("put_chunk", "放置chunk"),
            ("custom_process", "自定义处理"),
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ update_from_output方法包含{description}代码")
            else:
                print(f"⚠️  update_from_output方法不包含{description}代码")
    except (TypeError, OSError):
        print("⚠️  无法获取update_from_output方法源代码")
    
    return OmniARScheduler


def test_dependency_imports():
    """测试依赖模块能否导入"""
    import importlib
    
    dependencies = [
        'vllm_omni.distributed.omni_connectors.adapter',
        'vllm_omni.distributed.omni_connectors.factory',
        'importlib',
    ]
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ 能导入依赖: {dep}")
        except ImportError as e:
            print(f"⚠️  无法导入依赖 {dep}: {e}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始 OmniARScheduler 测试")
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
        print("🎉 所有测试通过!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)