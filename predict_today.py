#!/usr/bin/env python3
# 预测今天博时黄金C的涨跌

import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from socrates_system import SocratesSystem
    
    # 创建系统实例
    logger.info("创建苏格拉底预测系统实例...")
    socrates = SocratesSystem()
    
    # 爬取多源数据（包括博时黄金C真实数据）
    logger.info("爬取多源数据...")
    socrates.crawl_data()
    
    # 进行特征工程
    logger.info("进行特征工程...")
    socrates.feature_engineering()
    
    # 训练预测模型
    logger.info("训练预测模型...")
    socrates.train_predictors()
    
    # 进行今天的预测
    logger.info("进行今天的预测...")
    preds = socrates.make_predictions(forecast_steps=1)
    
    # 输出预测结果
    current_price = socrates.feature_matrix['fund_close'].iloc[-1]
    pred_price = preds[0]
    change = (pred_price - current_price) / current_price * 100
    trend = "上涨" if change > 0 else "下跌"
    
    print("\n" + "="*50)
    print("博时黄金C今日预测结果")
    print("="*50)
    print(f"当前价格: {current_price:.6f}")
    print(f"预测价格: {pred_price:.6f}")
    print(f"趋势: {trend} {abs(change):.2f}%")
    print("\n各传统方法预测:")
    
    # 输出各传统方法的预测结果
    if socrates.traditional_predictions:
        for method, result in socrates.traditional_predictions.items():
            method_name = {
                'i_ching': '易经',
                'liu_yao': '六爻',
                'qimen': '奇门遁甲',
                'tarot': '塔罗牌'
            }.get(method, method)
            print(f"{method_name}: {result['trend']} {result['change_percentage']:.2f}%")
    
    print("="*50)
    
except Exception as e:
    logger.error(f"预测过程中发生错误: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)