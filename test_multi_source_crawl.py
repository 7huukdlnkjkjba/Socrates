#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多源金融数据爬取测试脚本
用于验证黄金9999、伦敦金现和美元兑人民币汇率的数据爬取功能
"""

import logging
import pandas as pd
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据处理模块
from data_processing import DataProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gold9999_crawl():
    """测试黄金9999数据爬取"""
    logger.info("\n=== 测试黄金9999数据爬取 ===")
    
    processor = DataProcessor()
    try:
        # 爬取黄金9999数据（限制为50条记录）
        df = processor.crawl_gold9999(max_records=50)
        
        if df is not None and not df.empty:
            logger.info(f"✓ 成功爬取黄金9999数据，共{len(df)}条记录")
            logger.info(f"数据日期范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"数据列: {list(df.columns)}")
            logger.info("前5行数据:")
            logger.info(f"{df.head().to_string(index=False)}")
            
            # 检查数据类型
            logger.info("数据类型:")
            logger.info(f"{df.dtypes.to_string()}")
            
            return df
        else:
            logger.error("✗ 黄金9999数据爬取失败，返回空数据")
            return None
            
    except Exception as e:
        logger.error(f"✗ 黄金9999数据爬取发生错误: {str(e)}")
        return None

def test_gold_london_crawl():
    """测试伦敦金现数据爬取"""
    logger.info("\n=== 测试伦敦金现数据爬取 ===")
    
    processor = DataProcessor()
    try:
        # 爬取伦敦金现数据（限制为50条记录）
        df = processor.crawl_gold_london(max_records=50)
        
        if df is not None and not df.empty:
            logger.info(f"✓ 成功爬取伦敦金现数据，共{len(df)}条记录")
            logger.info(f"数据日期范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"数据列: {list(df.columns)}")
            logger.info("前5行数据:")
            logger.info(f"{df.head().to_string(index=False)}")
            
            # 检查数据类型
            logger.info("数据类型:")
            logger.info(f"{df.dtypes.to_string()}")
            
            return df
        else:
            logger.error("✗ 伦敦金现数据爬取失败，返回空数据")
            return None
            
    except Exception as e:
        logger.error(f"✗ 伦敦金现数据爬取发生错误: {str(e)}")
        return None

def test_usdcny_crawl():
    """测试美元兑人民币汇率数据爬取"""
    logger.info("\n=== 测试美元兑人民币汇率数据爬取 ===")
    
    processor = DataProcessor()
    try:
        # 爬取美元兑人民币汇率数据（限制为50条记录）
        df = processor.crawl_usdcny(max_records=50)
        
        if df is not None and not df.empty:
            logger.info(f"✓ 成功爬取美元兑人民币汇率数据，共{len(df)}条记录")
            logger.info(f"数据日期范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"数据列: {list(df.columns)}")
            logger.info("前5行数据:")
            logger.info(f"{df.head().to_string(index=False)}")
            
            # 检查数据类型
            logger.info("数据类型:")
            logger.info(f"{df.dtypes.to_string()}")
            
            return df
        else:
            logger.error("✗ 美元兑人民币汇率数据爬取失败，返回空数据")
            return None
            
    except Exception as e:
        logger.error(f"✗ 美元兑人民币汇率数据爬取发生错误: {str(e)}")
        return None

def test_data_integration():
    """测试多源数据集成"""
    logger.info("\n=== 测试多源数据集成 ===")
    
    # 爬取所有数据源
    gold9999_df = test_gold9999_crawl()
    gold_london_df = test_gold_london_crawl()
    usdcny_df = test_usdcny_crawl()
    
    if gold9999_df is not None and gold_london_df is not None and usdcny_df is not None:
        logger.info("\n=== 测试数据合并 ===")
        
        # 合并三个数据源
        merged_df = gold9999_df.merge(gold_london_df, on='date', suffixes=('_gold9999', '_gold_london'))
        merged_df = merged_df.merge(usdcny_df, on='date')
        
        logger.info(f"✓ 成功合并多源数据，共{len(merged_df)}条记录")
        logger.info(f"合并后的数据列: {list(merged_df.columns)}")
        logger.info("前5行合并数据:")
        logger.info(f"{merged_df.head().to_string(index=False)}")
        
        # 检查合并后的数据完整性
        logger.info(f"合并后数据日期范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        logger.info(f"缺失值情况:")
        logger.info(f"{merged_df.isnull().sum().to_string()}")
        
        return merged_df
    else:
        logger.error("✗ 多源数据集成失败，部分数据源爬取失败")
        return None

def main():
    """主测试函数"""
    logger.info("开始多源金融数据爬取测试")
    
    # 测试各个数据源的爬取
    gold9999_df = test_gold9999_crawl()
    gold_london_df = test_gold_london_crawl()
    usdcny_df = test_usdcny_crawl()
    
    # 测试数据集成
    merged_df = test_data_integration()
    
    logger.info("\n=== 测试总结 ===")
    logger.info(f"黄金9999数据爬取: {'成功' if gold9999_df is not None else '失败'}")
    logger.info(f"伦敦金现数据爬取: {'成功' if gold_london_df is not None else '失败'}")
    logger.info(f"美元兑人民币汇率数据爬取: {'成功' if usdcny_df is not None else '失败'}")
    logger.info(f"多源数据集成: {'成功' if merged_df is not None else '失败'}")
    
    if merged_df is not None:
        logger.info("\n测试成功完成！多源金融数据爬取功能正常工作。")
        return True
    else:
        logger.error("\n测试失败！请检查爬取功能。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
