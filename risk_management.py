import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RiskManagement:
    """风险管理模块
    
    实现风险指标计算、仓位管理、止损止盈等功能
    """
    
    def __init__(self, initial_capital=100000, risk_free_rate=0.02, max_position_size=0.1, stop_loss_pct=0.05, take_profit_pct=0.1):
        """初始化风险管理模块
        
        参数:
        initial_capital: 初始资金
        risk_free_rate: 无风险利率
        max_position_size: 最大仓位比例
        stop_loss_pct: 止损百分比
        take_profit_pct: 止盈百分比
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 交易记录
        self.trade_history = []
        # 仓位记录
        self.position_history = []
        # 风险指标记录
        self.risk_metrics_history = []
        
        logger.info(f"风险管理模块初始化完成，初始资金: {initial_capital}")
    
    def calculate_risk_metrics(self, returns):
        """计算风险指标
        
        参数:
        returns: 收益率序列
        
        返回:
        dict: 包含各种风险指标的字典
        """
        if len(returns) == 0:
            return {}
        
        # 确保使用真实的收益率序列，转换为numpy数组
        returns = np.array(returns)
        
        # 移除极端值（超过3个标准差的值）
        mean_return = returns.mean()
        std_return = returns.std()
        returns = returns[(returns >= mean_return - 3*std_return) & (returns <= mean_return + 3*std_return)]
        
        # 重新计算均值和标准差，确保波动率计算正确
        if len(returns) < 5:
            # 如果移除极端值后数据太少，使用原始数据
            returns = np.array(returns)
            mean_return = returns.mean()
            std_return = returns.std()
        
        metrics = {
            # 收益指标
            'total_return': returns.sum(),
            'annual_return': mean_return * 252,
            'daily_return': mean_return,
            'return_std': std_return,
            'annual_std': std_return * np.sqrt(252),
            
            # 风险指标 - 确保使用正确的无风险利率和收益率数据
            'sharpe_ratio': (mean_return - self.risk_free_rate/252) / std_return * np.sqrt(252) if std_return > 0 else 0,
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': (mean_return * 252) / self._calculate_max_drawdown(returns) if self._calculate_max_drawdown(returns) > 0 else 0,
            
            # 其他风险指标
            'value_at_risk_95': self._calculate_var(returns, 0.95),
            'value_at_risk_99': self._calculate_var(returns, 0.99),
            'conditional_var_95': self._calculate_cvar(returns, 0.95),
            'win_rate': self._calculate_win_rate(returns),
            'profit_factor': self._calculate_profit_factor(returns),
            'max_holding_period': self._calculate_max_holding_period(returns)
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns):
        """计算Sortino比率
        
        参数:
        returns: 收益率序列
        
        返回:
        float: Sortino比率
        """
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        
        excess_return = returns.mean() - self.risk_free_rate / 252
        return excess_return / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤
        
        参数:
        returns: 收益率序列（numpy数组或pandas Series）
        
        返回:
        float: 最大回撤
        """
        import pandas as pd
        cumulative_returns = (1 + returns).cumprod()
        
        # 将numpy数组转换为pandas Series，以便使用expanding方法
        if isinstance(cumulative_returns, np.ndarray):
            cumulative_returns = pd.Series(cumulative_returns)
        
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_var(self, returns, confidence_level=0.95):
        """计算Value at Risk (VaR)
        
        参数:
        returns: 收益率序列
        confidence_level: 置信水平
        
        返回:
        float: VaR值
        """
        if len(returns) == 0:
            return 0
        
        return -np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns, confidence_level=0.95):
        """计算Conditional Value at Risk (CVaR)
        
        参数:
        returns: 收益率序列
        confidence_level: 置信水平
        
        返回:
        float: CVaR值
        """
        if len(returns) == 0:
            return 0
        
        var = self._calculate_var(returns, confidence_level)
        return -returns[returns <= -var].mean()
    
    def _calculate_win_rate(self, returns):
        """计算胜率
        
        参数:
        returns: 收益率序列
        
        返回:
        float: 胜率
        """
        if len(returns) == 0:
            return 0
        
        winning_trades = len(returns[returns > 0])
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, returns):
        """计算盈利因子
        
        参数:
        returns: 收益率序列
        
        返回:
        float: 盈利因子
        """
        if len(returns) == 0:
            return 0
        
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())
        
        if total_loss == 0:
            return float('inf')
        
        return total_profit / total_loss
    
    def _calculate_max_holding_period(self, returns):
        """计算最大持仓周期
        
        参数:
        returns: 收益率序列
        
        返回:
        int: 最大持仓周期
        """
        # 简化实现，实际应该基于交易记录计算
        return len(returns)
    
    def calculate_position_size(self, current_price, volatility, risk_per_trade=0.02):
        """计算仓位大小
        
        参数:
        current_price: 当前价格
        volatility: 波动率
        risk_per_trade: 每笔交易的风险比例
        
        返回:
        float: 建议仓位大小
        """
        # 使用波动率调整仓位
        if volatility == 0:
            volatility = 0.01  # 默认波动率
        
        # 计算每单位风险
        risk_per_unit = current_price * volatility
        
        # 计算可承受的风险金额
        risk_amount = self.current_capital * risk_per_trade
        
        # 计算仓位数量
        position_size = risk_amount / risk_per_unit
        
        # 限制最大仓位
        max_position = self.current_capital * self.max_position_size / current_price
        position_size = min(position_size, max_position)
        
        # 确保仓位为正数
        position_size = max(0, position_size)
        
        return position_size
    
    def set_stop_loss_take_profit(self, entry_price, stop_loss_pct=None, take_profit_pct=None):
        """设置止损止盈
        
        参数:
        entry_price: 入场价格
        stop_loss_pct: 止损百分比
        take_profit_pct: 止盈百分比
        
        返回:
        dict: 包含止损和止盈价格的字典
        """
        stop_loss_pct = stop_loss_pct or self.stop_loss_pct
        take_profit_pct = take_profit_pct or self.take_profit_pct
        
        return {
            'stop_loss': entry_price * (1 - stop_loss_pct),
            'take_profit': entry_price * (1 + take_profit_pct)
        }
    
    def add_trade(self, trade_info):
        """添加交易记录
        
        参数:
        trade_info: 交易信息字典
        """
        self.trade_history.append(trade_info)
        logger.info(f"添加交易记录: {trade_info}")
    
    def add_position(self, position_info):
        """添加仓位记录
        
        参数:
        position_info: 仓位信息字典
        """
        self.position_history.append(position_info)
    
    def update_risk_metrics(self, metrics):
        """更新风险指标记录
        
        参数:
        metrics: 风险指标字典
        """
        self.risk_metrics_history.append(metrics)
    
    def get_current_risk_profile(self):
        """获取当前风险概况
        
        返回:
        dict: 当前风险概况
        """
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'trade_count': len(self.trade_history),
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }
    
    def check_model_drift(self, recent_returns, historical_returns, threshold=0.5):
        """检查模型漂移
        
        参数:
        recent_returns: 近期收益率
        historical_returns: 历史收益率
        threshold: 漂移阈值
        
        返回:
        bool: 是否发生模型漂移
        """
        if len(recent_returns) < 20 or len(historical_returns) < 50:
            return False
        
        # 计算均值差异
        recent_mean = recent_returns.mean()
        historical_mean = historical_returns.mean()
        
        # 计算标准差差异
        recent_std = recent_returns.std()
        historical_std = historical_returns.std()
        
        # 计算漂移指标
        mean_drift = abs(recent_mean - historical_mean) / historical_mean if historical_mean != 0 else 0
        std_drift = abs(recent_std - historical_std) / historical_std if historical_std != 0 else 0
        
        drift_score = (mean_drift + std_drift) / 2
        
        return drift_score > threshold
    
    def monitor_performance_decay(self, metrics_series, window=30, threshold=0.2):
        """监控性能衰减
        
        参数:
        metrics_series: 性能指标序列
        window: 监控窗口
        threshold: 衰减阈值
        
        返回:
        bool: 是否发生性能衰减
        """
        if len(metrics_series) < window * 2:
            return False
        
        # 计算近期和远期指标
        recent_metric = metrics_series[-window:].mean()
        long_term_metric = metrics_series[-window*2:-window].mean()
        
        # 计算衰减率
        if long_term_metric == 0:
            return False
        
        decay_rate = (long_term_metric - recent_metric) / long_term_metric
        
        return decay_rate > threshold
    
    def compare_volatility(self, actual_returns, predicted_volatility, threshold=0.5):
        """比较实际波动率与预测波动率
        
        参数:
        actual_returns: 实际收益率序列
        predicted_volatility: 预测波动率
        threshold: 波动率差异阈值
        
        返回:
        dict: 波动率比较结果
        """
        if len(actual_returns) == 0:
            return {
                'actual_volatility': 0,
                'predicted_volatility': predicted_volatility,
                'difference': 0,
                'warning': False,
                'message': '没有实际收益率数据，无法比较波动率'
            }
        
        # 计算实际波动率（日波动率）
        actual_volatility = actual_returns.std()
        
        # 计算年化波动率
        actual_volatility_annual = actual_volatility * np.sqrt(252)
        predicted_volatility_annual = predicted_volatility * np.sqrt(252) if predicted_volatility != 0 else 0
        
        # 计算波动率差异
        if actual_volatility != 0:
            volatility_diff = abs(actual_volatility - predicted_volatility) / actual_volatility
        else:
            volatility_diff = 0
        
        # 生成警告信息
        warning = False
        message = ""
        
        if volatility_diff > threshold:
            warning = True
            message = f"实际波动率({actual_volatility_annual:.4f})与预测波动率({predicted_volatility_annual:.4f})差异较大，差异率为{volatility_diff:.2%}，建议调整波动率预测模型"
            logger.warning(message)
        elif actual_volatility_annual < 0.05:  # 5%年化波动率视为异常低
            warning = True
            message = f"实际年化波动率({actual_volatility_annual:.4f})异常低，黄金基金波动率通常在10%-30%之间，可能存在数据质量问题"
            logger.warning(message)
        elif actual_volatility_annual > 0.5:  # 50%年化波动率视为异常高
            warning = True
            message = f"实际年化波动率({actual_volatility_annual:.4f})异常高，黄金基金波动率通常在10%-30%之间，可能存在极端市场情况或数据异常"
            logger.warning(message)
        else:
            message = f"实际年化波动率({actual_volatility_annual:.4f})在正常范围内，与预测波动率({predicted_volatility_annual:.4f})差异在可接受范围内"
            logger.info(message)
        
        return {
            'actual_volatility': actual_volatility,
            'actual_volatility_annual': actual_volatility_annual,
            'predicted_volatility': predicted_volatility,
            'predicted_volatility_annual': predicted_volatility_annual,
            'difference': volatility_diff,
            'warning': warning,
            'message': message
        }
    
    def simulate_trade(self, signal, current_price, date, volatility=0.01, transaction_cost=0.001):
        """模拟交易
        
        参数:
        signal: 交易信号 (1=买入, -1=卖出, 0=持有)
        current_price: 当前价格
        date: 交易日期
        volatility: 波动率
        transaction_cost: 交易成本
        
        返回:
        dict: 交易结果
        """
        trade_result = {
            'date': date,
            'signal': signal,
            'current_price': current_price,
            'volatility': volatility,
            'transaction_cost': transaction_cost,
            'action': 'hold'
        }
        
        # 计算建议仓位
        position_size = self.calculate_position_size(current_price, volatility)
        
        # 简单的交易逻辑
        if signal == 1 and position_size > 0:
            # 买入
            shares = position_size
            cost = shares * current_price * (1 + transaction_cost)
            if cost <= self.current_capital:
                self.current_capital -= cost
                trade_result['action'] = 'buy'
                trade_result['shares'] = shares
                trade_result['cost'] = cost
                
                # 设置止损止盈
                sl_tp = self.set_stop_loss_take_profit(current_price)
                trade_result['stop_loss'] = sl_tp['stop_loss']
                trade_result['take_profit'] = sl_tp['take_profit']
                
                # 添加交易记录
                self.add_trade({
                    'date': date,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'cost': cost,
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit': sl_tp['take_profit']
                })
        elif signal == -1:
            # 卖出（简化实现，实际应该基于当前持仓）
            trade_result['action'] = 'sell'
        
        # 添加仓位记录
        self.add_position({
            'date': date,
            'price': current_price,
            'position_size': position_size,
            'capital': self.current_capital
        })
        
        return trade_result
    
    def generate_risk_report(self):
        """生成风险报告
        
        返回:
        dict: 风险报告
        """
        # 计算收益率序列
        if len(self.position_history) < 2:
            return {
                'risk_profile': self.get_current_risk_profile(),
                'risk_metrics': {},
                'trade_summary': {
                    'total_trades': len(self.trade_history),
                    'winning_trades': 0,
                    'losing_trades': 0
                }
            }
        
        # 生成简单的收益率序列（基于仓位记录）
        prices = [pos['price'] for pos in self.position_history]
        returns = np.diff(prices) / prices[:-1]
        
        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(returns)
        
        # 计算交易统计
        winning_trades = len([t for t in self.trade_history if t.get('action') == 'buy'])
        losing_trades = len([t for t in self.trade_history if t.get('action') == 'sell'])
        
        return {
            'risk_profile': self.get_current_risk_profile(),
            'risk_metrics': risk_metrics,
            'trade_summary': {
                'total_trades': len(self.trade_history),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / len(self.trade_history) if len(self.trade_history) > 0 else 0
            }
        }
