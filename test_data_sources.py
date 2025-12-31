import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
import logging

# 设置日志级别为INFO，查看详细信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_all_data_sources():
    """测试所有数据源的爬取功能"""
    data_processor = DataProcessor()
    
    # 测试各个数据源
    data_types = ['gold9999', 'gold_london', 'usdcny', 'boshi_gold_c']
    
    for data_type in data_types:
        print(f"\n=== 测试数据源: {data_type} ===")
        try:
            # 爬取数据
            df = data_processor._crawl_financial_data(data_type, max_records=10)
            
            if df is not None and not df.empty:
                print(f"✓ 成功获取{data_type}数据")
                print(f"  数据量: {len(df)}条")
                print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
                print(f"  示例数据:")
                print(df.tail(3))
            else:
                print(f"✗ 获取{data_type}数据失败")
        except Exception as e:
            print(f"✗ 获取{data_type}数据时发生异常: {e}")

if __name__ == "__main__":
    test_all_data_sources()