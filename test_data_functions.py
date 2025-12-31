#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试每个数据获取函数
"""

import sys
import os
import logging
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor

def test_data_function(func_name, data_type, start_date=None, end_date=None, max_records=100):
    """
    测试单个数据获取函数
    
    参数:
    func_name: 函数名（用于日志）
    data_type: 数据类型
    start_date: 开始日期
    end_date: 结束日期
    max_records: 最大记录数
    """
    logger.info(f"=== 开始测试 {func_name} ({data_type}) ===")
    
    try:
        dp = DataProcessor()
        
        if hasattr(dp, func_name):
            # 调用指定的数据获取函数
            func = getattr(dp, func_name)
            df = func(start_date=start_date, end_date=end_date, max_records=max_records)
        else:
            # 直接调用通用的金融数据爬取函数
            df = dp._crawl_financial_data(data_type=data_type, max_records=max_records)
        
        # 验证返回的数据
        if df is not None and not df.empty:
            logger.info(f"✅ {func_name} ({data_type}) 成功获取数据")
            logger.info(f"   数据形状: {df.shape}")
            logger.info(f"   数据列: {list(df.columns)}")
            logger.info(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"   数据样例:\n{df.tail()}")
            return True, df
        else:
            logger.error(f"❌ {func_name} ({data_type}) 返回空数据")
            return False, None
    
    except Exception as e:
        logger.error(f"❌ {func_name} ({data_type}) 执行失败: {str(e)}")
        return False, None

if __name__ == "__main__":
    logger.info("开始测试所有数据获取函数...")
    
    # 测试每个数据类型
    data_types = [
        {'func_name': 'crawl_gold9999', 'data_type': 'gold9999', 'max_records': 20},
        {'func_name': 'crawl_gold_london', 'data_type': 'gold_london', 'max_records': 20},
        {'func_name': 'crawl_usdcny', 'data_type': 'usdcny', 'max_records': 20},
        {'func_name': 'crawl_boshi_gold_c', 'data_type': 'boshi_gold_c', 'max_records': 20}
    ]
    
    results = []
    
    for test_config in data_types:
        success, df = test_data_function(**test_config)
        results.append({
            'data_type': test_config['data_type'],
            'success': success,
            'data_shape': df.shape if df is not None else None
        })
        logger.info("="*50)
    
    # 总结测试结果
    logger.info("\n=== 测试结果总结 ===")
    for result in results:
        status = "✅" if result['success'] else "❌"
        data_shape = f"({result['data_shape'][0]}, {result['data_shape'][1]})" if result['data_shape'] else "None"
        logger.info(f"{status} {result['data_type']}: {'成功' if result['success'] else '失败'}, 数据形状: {data_shape}")
    
    # 统计成功率
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    logger.info(f"\n总体成功率: {success_count}/{total_count} ({(success_count/total_count)*100:.1f}%)")
