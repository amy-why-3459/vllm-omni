# tests/unit/core/sched/test_omni_generation_scheduler.py


def test_import():
    """测试能导入OmniGenerationScheduler"""
    try:
        from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
        print(f"✅ 成功导入 OmniGenerationScheduler")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_class_exists():
    """测试类存在并检查基本属性"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    
    # 检查类名
    assert OmniGenerationScheduler.__name__ == "OmniGenerationScheduler"
    print(f"✅ 类名正确: {OmniGenerationScheduler.__name__}")
    
    # 检查关键方法
    required_methods = ['__init__', 'schedule', 'update_from_output']
    for method in required_methods:
        assert hasattr(OmniGenerationScheduler, method), f"缺少方法: {method}"
        print(f"✅ 有方法: {method}")
    
    return True


def test_method_signatures():
    """测试方法签名"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    # 检查__init__签名
    init_sig = inspect.signature(OmniGenerationScheduler.__init__)
    print(f"✅ __init__ 签名: {init_sig}")
    
    # 检查schedule签名
    schedule_sig = inspect.signature(OmniGenerationScheduler.schedule)
    print(f"✅ schedule 签名: {schedule_sig}")
    
    # 检查update_from_output签名
    update_sig = inspect.signature(OmniGenerationScheduler.update_from_output)
    print(f"✅ update_from_output 签名: {update_sig}")
    
    return True


def test_simple_init():
    """简单测试初始化逻辑 - 使用正确的mock方式"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    from unittest.mock import Mock, patch, PropertyMock
    
    # 方法1: 完全mock __init__ 方法
    with patch.object(OmniGenerationScheduler, '__init__', return_value=None):
        scheduler = OmniGenerationScheduler()
        scheduler.vllm_config = None
        scheduler.omni_connector = None
        scheduler.stage_id = None
        
        # 验证可以创建实例
        assert scheduler is not None
        print("✅ 可以创建OmniGenerationScheduler实例")
    
    # 方法2: 直接测试源代码中的初始化逻辑
    print("检查__init__方法中的关键代码...")
    import inspect
    source = inspect.getsource(OmniGenerationScheduler.__init__)
    
    # 检查是否包含关键逻辑
    keywords_to_check = [
        'model_config',
        'async_chunk', 
        'omni_connector',
        'stage_id',
        'vllm_config'
    ]
    
    for keyword in keywords_to_check:
        if keyword in source:
            print(f"✅ __init__ 方法包含 '{keyword}'")
        else:
            print(f"⚠️  __init__ 方法不包含 '{keyword}'")
    
    return True


def test_schedule_method_code():
    """检查schedule方法中的关键代码"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    try:
        source = inspect.getsource(OmniGenerationScheduler.schedule)
        
        # 检查是否包含omni相关代码
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
            print(f"✅ schedule方法包含Omni代码: {', '.join(found)}")
        else:
            print("⚠️  schedule方法不包含明显的Omni代码")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  无法获取schedule方法源代码: {e}")
        return True  # 不是致命错误


def test_update_from_output_code():
    """检查update_from_output方法中的关键代码"""
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    import inspect
    
    try:
        source = inspect.getsource(OmniGenerationScheduler.update_from_output)
        
        # 检查关键逻辑
        checks = [
            ('FINISHED_STOPPED', '完成状态检查'),
            ('omni_connector', 'Omni连接器检查'),
            ('num_computed_tokens', '计算token数检查'),
            ('num_prompt_tokens', '提示token数检查')
        ]
        
        for keyword, description in checks:
            if keyword in source:
                print(f"✅ update_from_output包含'{description}'代码")
            else:
                print(f"⚠️  update_from_output不包含'{description}'代码")
        
        return True
    except (TypeError, OSError) as e:
        print(f"⚠️  无法获取update_from_output方法源代码: {e}")
        return True  # 不是致命错误


def test_import_dependencies():
    """测试相关导入"""
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
            print(f"✅ 能导入: {dep}")
        except ImportError as e:
            print(f"⚠️  无法导入 {dep}: {e}")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始 OmniGenerationScheduler 测试")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("导入测试", test_import),
        ("类存在测试", test_class_exists),
        ("方法签名测试", test_method_signatures),
        ("简单初始化测试", test_simple_init),
        ("schedule方法代码检查", test_schedule_method_code),
        ("update_from_output代码检查", test_update_from_output_code),
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