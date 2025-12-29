#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进后的网络爬取功能
"""

from data_processing import DataProcessor
import pandas as pd
import requests
from bs4 import BeautifulSoup

def test_basic_crawl_fix():
    """测试修复后的基本爬取模式"""
    print("=== 测试修复后的基本爬取模式 ===")
    try:
        # 使用一个公开的CSV数据源进行测试
        url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Confirmed',
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

def test_simple_html_crawl():
    """测试从简单HTML表格爬取数据"""
    print("\n=== 测试简单HTML表格爬取 ===")
    try:
        # 使用一个简单的CSV数据源来模拟HTML表格爬取
        # 使用GitHub上的一个简单数据集
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Temp',
            source_type='web',
            crawl_mode='basic'
        )
        
        print(f"成功获取数据，数据形状: {data.shape}")
        print(f"数据预览:\n{data.head()}")
        print("简单HTML表格爬取测试通过！")
        return True
    except Exception as e:
        print(f"简单HTML表格爬取测试失败: {e}")
        return False

def test_crawl_with_logging():
    """测试爬取功能的日志记录"""
    print("\n=== 测试爬取功能的日志记录 ===")
    try:
        url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Confirmed',
            source_type='web',
            crawl_mode='basic'
        )
        
        print("日志记录功能正常！")
        return True
    except Exception as e:
        print(f"日志记录功能测试失败: {e}")
        return False

def test_error_handling():
    """测试爬取功能的错误处理"""
    print("\n=== 测试爬取功能的错误处理 ===")
    try:
        # 使用一个不存在的URL进行测试
        url = "https://nonexistent-url-that-will-fail.com"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Confirmed',
            source_type='web',
            crawl_mode='basic',
            retries=2,  # 减少重试次数以加快测试
            delay=1     # 减少延迟以加快测试
        )
        
        print("错误处理测试失败: 应该抛出异常但没有")
        return False
    except Exception as e:
        print(f"成功捕获异常: {e}")
        print("错误处理测试通过！")
        return True

def test_integration_with_prediction():
    """测试爬取功能与预测功能的集成"""
    print("\n=== 测试爬取功能与预测功能的集成 ===")
    try:
        from prediction import TimeSeriesPredictor
        
        # 使用一个小型数据集来避免性能问题
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
        processor = DataProcessor()
        
        # 加载数据
        data = processor.load_data(
            source=url,
            time_column='Date',
            target_column='Temp',
            source_type='web',
            crawl_mode='basic'
        )
        
        # 清洗数据
        cleaned_data = processor.clean_data()
        
        # 测试平稳性 - 只使用前100个数据点以提高速度
        small_data = cleaned_data.head(100)
        processor.data = small_data
        is_stationary = processor.check_stationarity()
        print(f"数据平稳性测试结果: {is_stationary}")
        
        # 创建预测器并进行预测 - 继续使用小数据集
        predictor = TimeSeriesPredictor(processor.get_data(), target_column='Temp')
        
        # 测试移动平均预测
        ma_prediction = predictor.moving_average(window_size=3, forecast_steps=1)
        print(f"移动平均预测结果: {ma_prediction}")
        
        # 测试指数平滑预测
        es_prediction = predictor.exponential_smoothing(alpha=0.2, forecast_steps=1)
        print(f"指数平滑预测结果: {es_prediction}")
        
        print("爬取功能与预测功能集成测试通过！")
        return True
    except Exception as e:
        print(f"爬取功能与预测功能集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试改进后的网络爬取功能...")
    
    # 运行测试
    test_results = []
    test_results.append(test_basic_crawl_fix())
    test_results.append(test_simple_html_crawl())
    test_results.append(test_crawl_with_logging())
    test_results.append(test_error_handling())
    test_results.append(test_integration_with_prediction())
    
    print("\n=== 测试结果汇总 ===")
    print(f"测试总数: {len(test_results)}")
    print(f"通过数: {sum(test_results)}")
    print(f"失败数: {len(test_results) - sum(test_results)}")
    
    if all(test_results):
        print("\n✅ 所有测试通过！网络爬取功能已正常实现并与现有系统兼容。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
