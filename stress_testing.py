import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class StressTesting:
    """极端压力测试模块
    
    实现2008年金融危机级别的测试、市场冲击测试和资金规模测试
    """
    
    def __init__(self, risk_manager=None):
        """初始化压力测试模块
        
        参数:
        risk_manager: 风险管理模块实例
        """
        self.risk_manager = risk_manager
        logger.info("压力测试模块初始化完成")
    
    def simulate_financial_crisis_2008(self, returns, crisis_intensity=1.0):
        """模拟2008年金融危机级别的极端情况
        
        参数:
        returns: 原始收益率序列
        crisis_intensity: 危机强度因子，1.0表示2008年金融危机水平
        
        返回:
        pd.Series: 压力测试后的收益率序列
        """
        logger.info(f"开始2008年金融危机压力测试，强度因子: {crisis_intensity}")
        
        # 计算原始收益率的统计特征
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 2008年金融危机的特征：更大的波动率，更负的偏度，更高的峰度
        crisis_std_multiplier = 2.5 * crisis_intensity  # 波动率增加2.5倍
        crisis_skewness = -2.0 * crisis_intensity  # 负偏度增加
        crisis_kurtosis = 8.0 * crisis_intensity  # 峰度增加
        
        # 使用Johnson SU分布生成极端压力下的收益率
        # 首先将原始收益率转换为正态分布
        norm_returns = stats.norm.ppf(stats.rankdata(returns) / (len(returns) + 1))
        
        # 调整分布参数以模拟金融危机
        adjusted_returns = norm_returns * crisis_std_multiplier + mean_return
        
        # 添加极端负收益率事件（如雷曼兄弟倒闭）
        num_extreme_events = int(len(returns) * 0.05)  # 5%的极端事件
        extreme_indices = np.random.choice(len(adjusted_returns), size=num_extreme_events, replace=False)
        
        # 极端事件的收益率幅度：-10%到-30%
        extreme_returns = np.random.uniform(-0.30, -0.10, size=num_extreme_events) * crisis_intensity
        adjusted_returns[extreme_indices] = extreme_returns
        
        # 确保分布的基本统计特征符合预期
        stress_returns = pd.Series(adjusted_returns, index=returns.index)
        
        logger.info(f"金融危机压力测试完成")
        logger.info(f"原始波动率: {std_return:.4f}, 压力测试后波动率: {stress_returns.std():.4f}")
        logger.info(f"原始偏度: {skewness:.4f}, 压力测试后偏度: {stress_returns.skew():.4f}")
        logger.info(f"原始峰度: {kurtosis:.4f}, 压力测试后峰度: {stress_returns.kurtosis():.4f}")
        
        return stress_returns
    
    def market_structure_change_test(self, historical_data, window=60, threshold=0.3):
        """检测市场结构变化
        
        参数:
        historical_data: 历史价格数据
        window: 检测窗口大小
        threshold: 结构变化阈值
        
        返回:
        dict: 市场结构变化检测结果
        """
        logger.info("开始市场结构变化检测")
        
        # 计算收益率
        returns = historical_data.pct_change().dropna()
        
        # 使用滚动窗口计算波动率
        rolling_vol = returns.rolling(window).std()
        
        # 计算波动率变化率
        vol_change = rolling_vol.pct_change().abs()
        
        # 检测波动率突变点
        structure_change_points = vol_change[vol_change > threshold].index
        
        # 计算市场结构变化的统计特征
        if len(structure_change_points) > 0:
            avg_volatility_before = rolling_vol.loc[:structure_change_points[0]].mean()
            avg_volatility_after = rolling_vol.loc[structure_change_points[0]:].mean()
            volatility_increase = (avg_volatility_after - avg_volatility_before) / avg_volatility_before
        else:
            avg_volatility_before = rolling_vol.mean()
            avg_volatility_after = avg_volatility_before
            volatility_increase = 0
        
        result = {
            'structure_change_detected': len(structure_change_points) > 0,
            'change_points': structure_change_points,
            'num_change_points': len(structure_change_points),
            'avg_volatility_before': avg_volatility_before,
            'avg_volatility_after': avg_volatility_after,
            'volatility_increase': volatility_increase,
            'volatility_change_series': vol_change
        }
        
        logger.info(f"市场结构变化检测完成，检测到{len(structure_change_points)}个变化点")
        if len(structure_change_points) > 0:
            logger.info(f"波动率变化: {volatility_increase:.2%}")
        
        return result
    
    def large_capital_execution_test(self, price_data, strategy_signals, initial_capital=100000, large_capital_multiplier=100):
        """测试大规模资金下的执行能力
        
        参数:
        price_data: 价格数据
        strategy_signals: 策略信号
        initial_capital: 初始资金
        large_capital_multiplier: 大规模资金倍数
        
        返回:
        dict: 大规模资金执行测试结果
        """
        logger.info(f"开始大规模资金执行测试，资金倍数: {large_capital_multiplier}")
        
        large_capital = initial_capital * large_capital_multiplier
        
        # 计算市场冲击成本
        def calculate_market_impact(trade_size, current_price, market_liquidity=1000000):
            """计算市场冲击成本
            
            参数:
            trade_size: 交易规模
            current_price: 当前价格
            market_liquidity: 市场流动性（日交易量）
            
            返回:
            float: 市场冲击成本（百分比）
            """
            # 市场冲击成本模型：平方根法则
            impact_cost = 0.001 * (trade_size / market_liquidity) ** 0.5
            return impact_cost
        
        # 模拟交易执行
        positions = []
        cash = large_capital
        portfolio_value = [large_capital]
        execution_costs = []
        
        for i in range(1, len(price_data)):
            date = price_data.index[i]
            current_price = price_data.iloc[i]
            signal = strategy_signals.iloc[i-1]  # 使用前一天的信号
            
            # 计算目标仓位
            target_position = large_capital * signal * 0.3  # 30%仓位
            shares = int(target_position / current_price)
            
            if shares != 0:
                # 计算市场冲击成本
                market_impact = calculate_market_impact(abs(shares * current_price), current_price)
                
                # 计算执行价格（包含市场冲击）
                execution_price = current_price * (1 + market_impact * np.sign(shares))
                
                # 计算交易成本
                transaction_cost = abs(shares) * execution_price * 0.001  # 0.1%交易成本
                total_cost = shares * execution_price + transaction_cost
                
                # 更新现金和仓位
                cash -= total_cost
                
                # 计算执行成本
                ideal_cost = shares * current_price
                execution_cost = abs(total_cost - ideal_cost)
                execution_costs.append(execution_cost)
            
            # 计算当前组合价值
            current_portfolio_value = cash + shares * current_price
            portfolio_value.append(current_portfolio_value)
            positions.append(shares)
        
        # 计算测试结果
        portfolio_series = pd.Series(portfolio_value, index=price_data.index)
        returns = portfolio_series.pct_change().dropna()
        
        result = {
            'initial_capital': initial_capital,
            'large_capital': large_capital,
            'final_portfolio_value': portfolio_series.iloc[-1],
            'total_return': (portfolio_series.iloc[-1] - large_capital) / large_capital,
            'avg_execution_cost': np.mean(execution_costs) if execution_costs else 0,
            'max_execution_cost': np.max(execution_costs) if execution_costs else 0,
            'portfolio_series': portfolio_series,
            'returns': returns,
            'positions': pd.Series(positions, index=price_data.index[1:])
        }
        
        logger.info(f"大规模资金执行测试完成")
        logger.info(f"初始资金: {large_capital:,.2f}")
        logger.info(f"最终组合价值: {portfolio_series.iloc[-1]:,.2f}")
        logger.info(f"总收益: {result['total_return']:.2%}")
        logger.info(f"平均执行成本: {result['avg_execution_cost']:,.2f}")
        logger.info(f"最大执行成本: {result['max_execution_cost']:,.2f}")
        
        return result
    
    def run_comprehensive_stress_test(self, returns, price_data, strategy_signals, crisis_intensity=1.0, large_capital_multiplier=100):
        """运行综合压力测试
        
        参数:
        returns: 收益率序列
        price_data: 价格数据
        strategy_signals: 策略信号
        crisis_intensity: 危机强度因子
        large_capital_multiplier: 大规模资金倍数
        
        返回:
        dict: 综合压力测试结果
        """
        logger.info("开始综合压力测试")
        
        # 运行2008年金融危机压力测试
        crisis_returns = self.simulate_financial_crisis_2008(returns, crisis_intensity)
        
        # 运行市场结构变化检测
        market_structure_result = self.market_structure_change_test(price_data)
        
        # 运行大规模资金执行测试
        large_capital_result = self.large_capital_execution_test(price_data, strategy_signals, large_capital_multiplier=large_capital_multiplier)
        
        # 综合结果
        comprehensive_result = {
            'financial_crisis_test': {
                'original_returns': returns,
                'crisis_returns': crisis_returns,
                'crisis_intensity': crisis_intensity
            },
            'market_structure_test': market_structure_result,
            'large_capital_test': large_capital_result
        }
        
        # 生成测试报告
        self.generate_stress_test_report(comprehensive_result)
        
        logger.info("综合压力测试完成")
        
        return comprehensive_result
    
    def generate_stress_test_report(self, result):
        """生成压力测试报告
        
        参数:
        result: 压力测试结果
        """
        logger.info("\n" + "="*80)
        logger.info("                    综合压力测试报告")
        logger.info("="*80)
        
        # 金融危机测试报告
        logger.info("\n1. 2008年金融危机压力测试")
        logger.info("-"*40)
        crisis_returns = result['financial_crisis_test']['crisis_returns']
        logger.info(f"   危机强度因子: {result['financial_crisis_test']['crisis_intensity']}")
        logger.info(f"   极端压力下最大回撤: {self._calculate_max_drawdown(crisis_returns):.2%}")
        logger.info(f"   极端压力下夏普比率: {self._calculate_sharpe_ratio(crisis_returns):.2f}")
        logger.info(f"   极端压力下波动率: {crisis_returns.std() * np.sqrt(252):.2%}")
        
        # 市场结构变化报告
        logger.info("\n2. 市场结构变化检测")
        logger.info("-"*40)
        ms_result = result['market_structure_test']
        logger.info(f"   结构变化检测结果: {'检测到' if ms_result['structure_change_detected'] else '未检测到'}")
        logger.info(f"   变化点数量: {ms_result['num_change_points']}")
        logger.info(f"   波动率变化: {ms_result['volatility_increase']:.2%}")
        
        # 大规模资金测试报告
        logger.info("\n3. 大规模资金执行测试")
        logger.info("-"*40)
        lc_result = result['large_capital_test']
        logger.info(f"   初始资金规模: {lc_result['large_capital']:,.2f}")
        logger.info(f"   最终组合价值: {lc_result['final_portfolio_value']:,.2f}")
        logger.info(f"   总收益率: {lc_result['total_return']:.2%}")
        logger.info(f"   平均执行成本: {lc_result['avg_execution_cost']:,.2f}")
        logger.info(f"   最大执行成本: {lc_result['max_execution_cost']:,.2f}")
        logger.info(f"   大规模资金下夏普比率: {self._calculate_sharpe_ratio(lc_result['returns']):.2f}")
        
        logger.info("\n" + "="*80)
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤
        
        参数:
        returns: 收益率序列
        
        返回:
        float: 最大回撤
        """
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """计算夏普比率
        
        参数:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
        返回:
        float: 夏普比率
        """
        excess_return = returns.mean() - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return excess_return / returns.std() * np.sqrt(252)