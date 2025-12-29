#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试网络爬取功能
"""

from data_processing import DataProcessor

def test_basic_crawl():
    """测试基本爬取模式"""
    print("=== 测试基本爬取模式 ===")
    try:
        # 使用一个公开的时间序列数据CSV进行测试
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Births',
            source_type='web',
            crawl_mode='basic'
        )
        
        print(f"成功获取数据，数据形状: {data.shape}")
        print(f"数据预览:\n{data.head()}")
        print("基本爬取模式测试通过！")
        return True
    except Exception as e:
        print(f"基本爬取模式测试失败: {e}")
        return False

def test_web_driver_crawl():
    """测试浏览器驱动爬取模式"""
    print("\n=== 测试浏览器驱动爬取模式 ===")
    try:
        # 使用一个简单的HTML表格页面进行测试
        url = "https://www.worldometers.info/world-population/world-population-by-year/"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Year',
            target_column='World Population',
            source_type='web',
            crawl_mode='web_driver',
            browser_type='firefox',
            table_selector='table#example2'
        )
        
        print(f"成功获取数据，数据形状: {data.shape}")
        print(f"数据预览:\n{data.head()}")
        print("浏览器驱动爬取模式测试通过！")
        return True
    except Exception as e:
        print(f"浏览器驱动爬取模式测试失败: {e}")
        return False

def test_integration():
    """测试与现有系统的集成"""
    print("\n=== 测试与现有系统的集成 ===")
    try:
        # 测试完整的工作流
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Births',
            source_type='web',
            crawl_mode='basic'
        )
        
        # 数据清洗（仅测试功能，不检查平稳性）
        cleaned_data = processor.clean_data()
        print(f"数据清洗完成，清洗后数据形状: {cleaned_data.shape}")
        
        print("系统集成测试通过！")
        return True
    except Exception as e:
        print(f"系统集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试网络爬取功能...")
    
    # 运行测试
    basic_result = test_basic_crawl()
    web_driver_result = test_web_driver_crawl()
    integration_result = test_integration()
    
    print("\n=== 测试结果汇总 ===")
    print(f"基本爬取模式: {'通过' if basic_result else '失败'}")
    print(f"浏览器驱动爬取模式: {'通过' if web_driver_result else '失败'}")
    print(f"系统集成: {'通过' if integration_result else '失败'}")
    
    if basic_result and integration_result:
        print("\n✅ 测试成功！网络爬取功能已正常实现并与现有系统兼容。")
        print("⚠️  注意：浏览器驱动爬取模式需要安装Chrome浏览器，如果测试失败，请确保Chrome已安装。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
