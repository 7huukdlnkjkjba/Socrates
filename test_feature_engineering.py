import pandas as pd
import logging
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """测试特征工程功能"""
    logger.info("=== 测试特征工程功能 ===")
    
    try:
        # 1. 创建DataProcessor实例
        processor = DataProcessor()
        
        # 2. 爬取多源数据
        logger.info("开始爬取多源数据")
        gold9999_data = processor.crawl_gold9999()
        gold_london_data = processor.crawl_gold_london()
        usdcny_data = processor.crawl_usdcny()
        
        # 3. 生成模拟的基金数据（因为实际爬取可能受限）
        logger.info("生成模拟基金数据")
        # 使用与其他数据源相同的日期范围
        dates = gold9999_data['date']
        
        # 创建模拟的基金数据
        import numpy as np
        np.random.seed(42)
        
        # 生成模拟的基金净值（与黄金价格有一定相关性）
        gold9999_close = gold9999_data['close'].values
        # 添加一些噪声，使得基金净值与黄金价格有一定相关性但不完全相同
        fund_close = 1.5 + 0.001 * gold9999_close + np.random.normal(0, 0.02, len(gold9999_close))
        
        # 创建基金数据DataFrame
        fund_data = pd.DataFrame({
            'date': dates,
            'open': fund_close * np.random.uniform(0.995, 1.005, len(fund_close)),
            'high': fund_close * np.random.uniform(1.0, 1.01, len(fund_close)),
            'low': fund_close * np.random.uniform(0.99, 1.0, len(fund_close)),
            'close': fund_close,
            'volume': np.random.randint(100000, 1000000, len(fund_close))
        })
        
        logger.info(f"生成的基金数据：{len(fund_data)}条记录")
        logger.info(f"数据日期范围: {fund_data['date'].min()} 到 {fund_data['date'].max()}")
        
        # 4. 创建FeatureEngineer实例
        feature_engineer = FeatureEngineer()
        
        # 5. 加载数据
        logger.info("加载数据到特征工程模块")
        feature_engineer.load_data(fund_data, gold9999_data, gold_london_data, usdcny_data)
        
        # 6. 创建特征矩阵
        logger.info("创建特征矩阵")
        feature_matrix = feature_engineer.create_feature_matrix(
            rolling_correlation_windows=[10, 20],
            momentum_windows=[5, 10, 20],
            volatility_windows=[10, 20, 30]
        )
        
        # 7. 验证特征矩阵
        logger.info(f"特征矩阵创建成功，形状: {feature_matrix.shape}")
        logger.info(f"特征数量: {len(feature_matrix.columns)}")
        logger.info(f"特征矩阵日期范围: {feature_matrix.index.min()} 到 {feature_matrix.index.max()}")
        
        # 打印前5行特征矩阵
        logger.info("\n前5行特征矩阵:")
        logger.info(feature_matrix.head().to_string())
        
        # 8. 计算特征重要性
        logger.info("\n计算特征重要性")
        feature_importance = feature_engineer.get_feature_importance(target_column='fund_close')
        logger.info("\n特征重要性前10名:")
        logger.info(feature_importance.head(10).to_string())
        
        # 9. 检查特征是否包含预期的内容
        expected_features = [
            'corr_fund_gold9999_10d',
            'corr_fund_gold_london_20d',
            'gold_price_spread',
            'theoretical_nav',
            'fund_momentum_5d',
            'fund_volatility_20d',
            'fund_rsi_14d',
            'fund_bollinger_width_20d',
            'fund_close',
            'gold9999_close',
            'gold_london_close',
            'usdcny_close'
        ]
        
        logger.info("\n验证预期特征是否存在:")
        for feature in expected_features:
            if feature in feature_matrix.columns:
                logger.info(f"✓ {feature} 存在")
            else:
                logger.warning(f"✗ {feature} 不存在")
        
        # 10. 保存特征矩阵到文件
        feature_matrix.to_csv('feature_matrix_test.csv')
        logger.info("\n特征矩阵已保存到 feature_matrix_test.csv")
        
        logger.info("\n=== 特征工程功能测试成功完成！===")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_feature_engineering()
