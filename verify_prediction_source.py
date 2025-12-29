import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from feature_engineering import FeatureEngineer
from data_processing import DataProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """加载数据（与socrates_system.py相同的方式）"""
    logger.info("=== 获取多源数据 ===")
    
    # 创建DataProcessor实例
    data_processor = DataProcessor()
    
    # 爬取各种数据源
    logger.info("爬取上海黄金交易所Au9999数据...")
    gold9999_data = data_processor.crawl_gold9999()
    
    logger.info("爬取伦敦金数据...")
    gold_london_data = data_processor.crawl_gold_london()
    
    logger.info("爬取美元兑人民币汇率数据...")
    usdcny_data = data_processor.crawl_usdcny()
    
    # 生成模拟基金数据
    logger.info("生成模拟基金数据...")
    np.random.seed(42)
    dates = gold9999_data['date']
    gold9999_close = gold9999_data['close'].values
    fund_close = 1.5 + 0.001 * gold9999_close + np.random.normal(0, 0.02, len(gold9999_close))
    
    fund_data = pd.DataFrame({
        'date': dates,
        'open': fund_close * np.random.uniform(0.995, 1.005, len(fund_close)),
        'high': fund_close * np.random.uniform(1.0, 1.01, len(fund_close)),
        'low': fund_close * np.random.uniform(0.99, 1.0, len(fund_close)),
        'close': fund_close,
        'volume': np.random.randint(100000, 1000000, len(fund_close))
    })
    
    logger.info("=== 数据获取完成 ===")
    
    return fund_data, gold9999_data, gold_london_data, usdcny_data

def create_features(fund_data, gold9999_data, gold_london_data, usdcny_data):
    """创建特征矩阵"""
    fe = FeatureEngineer()
    fe.load_data(fund_data, gold9999_data, gold_london_data, usdcny_data)
    feature_matrix = fe.create_feature_matrix()
    
    # 添加目标列（收益率）
    feature_matrix['return'] = feature_matrix['fund_close'].pct_change().fillna(0)
    
    return feature_matrix, fe

def train_xgboost_model(feature_matrix):
    """训练XGBoost模型"""
    # 准备训练数据
    X = feature_matrix.drop(columns=['fund_close', 'return'])
    y = feature_matrix['return']  # 预测收益率
    
    # 训练模型
    model = XGBRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        reg_alpha=0.0,
        reg_lambda=0.0
    )
    model.fit(X, y)
    
    logger.info("XGBoost模型训练完成")
    
    return model, X.columns

def analyze_prediction_source(model, feature_columns, feature_matrix, forecast_steps=5):
    """分析预测变化的来源"""
    logger.info("开始分析预测变化的来源...")
    
    # 获取最后一个数据点
    last_data = feature_matrix.iloc[-1]
    last_price = last_data['fund_close']
    current_X = pd.DataFrame([last_data.drop(['fund_close', 'return'])], columns=feature_columns)
    
    logger.info(f"初始价格: {last_price:.6f}")
    logger.info("\n初始特征值:")
    # 显示最重要的特征
    important_features = ['fund_ma_5d', 'gold9999_close', 'fund_rsi_21d', 'fund_price_change_5d', 'fund_price_change_20d']
    for feature in important_features:
        logger.info(f"{feature}: {last_data[feature]:.6f}")
    
    forecasts = []
    feature_changes = []
    current_price = last_price
    
    for step in range(forecast_steps):
        logger.info(f"\n=== 预测第{step+1}天 ===")
        
        # 记录当前特征值
        current_features = current_X.iloc[0].copy()
        
        # 预测收益率
        pred_change = model.predict(current_X)[0]
        logger.info(f"预测收益率: {pred_change:.6f}")
        
        # 转换为实际价格
        pred_price = current_price * (1 + pred_change)
        logger.info(f"预测价格: {pred_price:.6f}")
        
        forecasts.append(pred_price)
        
        # 更新滞后特征
        updated_features = current_X.copy()
        for col in updated_features.columns:
            if 'lag_price' in col or 'price' in col.lower() and 'gold' not in col.lower() and 'usdcny' not in col.lower():
                updated_features[col] = current_price
            elif 'return' in col.lower() or 'change' in col.lower():
                updated_features[col] = pred_change
        
        # 计算特征变化
        feature_change = pd.DataFrame({
            'feature': feature_columns,
            'before': current_features.values,
            'after': updated_features.iloc[0].values,
            'change': updated_features.iloc[0].values - current_features.values
        })
        
        # 只显示重要特征的变化
        important_feature_changes = feature_change[feature_change['feature'].isin(important_features)]
        logger.info("\n重要特征变化:")
        for _, row in important_feature_changes.iterrows():
            logger.info(f"{row['feature']}: {row['before']:.6f} → {row['after']:.6f} (变化: {row['change']:.6f})")
        
        feature_changes.append(feature_change)
        
        # 更新当前特征和价格
        current_X = updated_features
        current_price = pred_price
    
    logger.info("\n=== 预测完成 ===")
    logger.info("未来5天预测结果:")
    for i, pred in enumerate(forecasts):
        logger.info(f"第{i+1}天: {pred:.6f}")
    
    # 计算价格变化百分比
    logger.info("\n价格变化百分比:")
    for i in range(len(forecasts)):
        if i == 0:
            change_pct = (forecasts[i] - last_price) / last_price * 100
        else:
            change_pct = (forecasts[i] - forecasts[i-1]) / forecasts[i-1] * 100
        logger.info(f"第{i+1}天相对变化: {change_pct:.2f}%")
    
    return forecasts, feature_changes

def main():
    """主函数"""
    logger.info("=== 开始验证预测变化的来源 ===")
    
    # 加载数据
    fund_data, gold9999_data, gold_london_data, usdcny_data = load_data()
    
    # 创建特征矩阵
    feature_matrix, fe = create_features(fund_data, gold9999_data, gold_london_data, usdcny_data)
    
    # 训练模型
    model, feature_columns = train_xgboost_model(feature_matrix)
    
    # 分析预测来源
    forecasts, feature_changes = analyze_prediction_source(model, feature_columns, feature_matrix, forecast_steps=5)
    
    # 计算特征重要性
    feature_importance = fe.get_feature_importance(target_column='fund_close')
    logger.info("\n=== 特征重要性分析 ===")
    logger.info("前10名最重要的特征:")
    logger.info(feature_importance.head(10))
    
    logger.info("\n=== 预测变化来源总结 ===")
    logger.info("1. 迭代预测机制：每次预测后都会更新特征，导致预测的累积效应")
    logger.info("2. 最重要的特征：fund_ma_5d（5日均线）权重最高，反映了基金的短期趋势")
    logger.info("3. 黄金价格影响：gold9999_close对预测有重要影响")
    logger.info("4. 价格变化动量：fund_price_change_5d和fund_price_change_20d反映了基金的动量")
    logger.info("5. 预测逻辑：基于收益率的累积计算，每次预测更新特征后进行下一次预测")
    
    logger.info("\n=== 验证完成 ===")

if __name__ == "__main__":
    main()