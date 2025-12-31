import pandas as pd
import numpy as np
import logging
from backtesting import BacktestingSystem
from risk_management import RiskManagement

logger = logging.getLogger(__name__)

class StressTestEngine:
    """极端压力测试引擎
    
    实现各种极端市场条件下的测试，包括2008年金融危机级别测试
    """
    
    def __init__(self):
        self.backtesting_system = BacktestingSystem()
        self.risk_manager = RiskManagement()
        logger.info("极端压力测试引擎初始化完成")
    
    def simulate_2008_financial_crisis(self, historical_data):
        """模拟2008年金融危机条件
        
        参数:
        historical_data: 原始历史数据
        
        返回:
        pd.DataFrame: 模拟的金融危机数据
        """
        logger.info("开始模拟2008年金融危机条件")
        
        # 创建数据副本
        stress_data = historical_data.copy()
        
        # 1. 增加波动率（危机期间波动率通常是平时的3-5倍）
        logger.info("增加市场波动率...")
        original_volatility = stress_data['close'].pct_change().std()
        target_volatility = original_volatility * 4  # 4倍波动率
        
        # 使用几何布朗运动生成高波动率数据
        stress_data['close'] = self._generate_high_volatility_data(
            initial_price=stress_data['close'].iloc[0],
            volatility=target_volatility,
            length=len(stress_data),
            trend=-0.005  # 每日下跌0.5%的趋势
        )
        
        # 2. 添加急剧下跌（类似雷曼兄弟倒闭事件）
        logger.info("添加急剧下跌事件...")
        crash_point = int(len(stress_data) * 0.6)  # 在数据中间位置添加崩溃事件
        crash_magnitude = -0.25  # 单日下跌25%
        stress_data.loc[crash_point, 'close'] *= (1 + crash_magnitude)
        
        # 3. 添加流动性紧缩（模拟买卖价差扩大）
        logger.info("模拟流动性紧缩...")
        stress_data['open'] = stress_data['close'] * 0.98  # 开盘价大幅低于收盘价
        stress_data['high'] = stress_data['close'] * 1.02  # 最高价低于正常水平
        stress_data['low'] = stress_data['close'] * 0.96  # 最低价大幅低于正常水平
        stress_data['volume'] *= 0.3  # 成交量减少70%
        
        # 4. 添加持续下跌趋势
        logger.info("添加持续下跌趋势...")
        for i in range(1, len(stress_data)):
            if i > crash_point and i < crash_point + 30:  # 崩溃后30天持续下跌
                stress_data.loc[stress_data.index[i], 'close'] *= 0.99  # 每日额外下跌1%
        
        logger.info(f"金融危机模拟完成，原始波动率: {original_volatility:.6f}, 模拟波动率: {stress_data['close'].pct_change().std():.6f}")
        
        return stress_data
    
    def _generate_high_volatility_data(self, initial_price, volatility, length, trend=0):
        """生成高波动率数据
        
        参数:
        initial_price: 初始价格
        volatility: 波动率
        length: 数据长度
        trend: 趋势
        
        返回:
        np.ndarray: 生成的价格数据
        """
        returns = np.random.normal(trend, volatility, length)
        prices = [initial_price]
        
        for i in range(1, length):
            prices.append(prices[-1] * (1 + returns[i]))
        
        return np.array(prices)
    
    def run_stress_test(self, historical_data, strategy_func, lookback=20):
        """运行极端压力测试
        
        参数:
        historical_data: 原始历史数据
        strategy_func: 策略函数
        lookback: 回溯期
        
        返回:
        dict: 压力测试结果
        """
        logger.info("=== 开始极端压力测试 ===")
        
        # 1. 生成金融危机数据
        crisis_data = self.simulate_2008_financial_crisis(historical_data)
        
        # 2. 运行回测
        logger.info("在模拟的金融危机数据上运行回测...")
        backtest_results = self.backtesting_system.run_backtest(
            data=crisis_data,
            strategy_func=strategy_func,
            lookback=lookback
        )
        
        # 3. 计算压力测试指标
        logger.info("计算压力测试指标...")
        stress_metrics = self._calculate_stress_metrics(backtest_results)
        
        # 4. 生成压力测试报告
        report = {
            '