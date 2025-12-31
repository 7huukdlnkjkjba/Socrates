import pandas as pd
import numpy as np
import logging
from risk_management import RiskManagement

logger = logging.getLogger(__name__)

class BacktestingSystem:
    """回测系统
    
    实现历史数据回测、多种回测指标计算、市场机制和交易成本模拟
    """
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001, slippage=0.0005, commission=0.0003, market_impact=0.001):
        """初始化回测系统
        
        参数:
        initial_capital: 初始资金
        transaction_cost: 交易成本
        slippage: 滑点
        commission: 佣金
        market_impact: 市场冲击成本
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.commission = commission
        self.market_impact = market_impact
        
        # 回测结果
        self.backtest_results = {}
        # 回测指标
        self.backtest_metrics = {}
        # 交易记录
        self.backtest_trades = []
        # 仓位记录
        self.backtest_positions = []
        # 净值曲线
        self.equity_curve = []
        # 日收益率序列
        self.daily_returns = []
        
        # 风险管理模块
        self.risk_manager = RiskManagement(initial_capital=initial_capital)
        
        logger.info(f"回测系统初始化完成，初始资金: {initial_capital}")
    
    def run_backtest(self, data, strategy_func, lookback=20, forecast_steps=1):
        """运行回测
        
        参数:
        data: 历史数据
        strategy_func: 策略函数
        lookback: 回溯期
        forecast_steps: 预测步长
        
        返回:
        dict: 回测结果
        """
        logger.info(f"开始回测，数据长度: {len(data)}, 回溯期: {lookback}")
        
        # 初始化
        self.reset()
        
        # 净值曲线初始化
        self.equity_curve.append({
            'date': data['date'].iloc[lookback-1],
            'equity': self.current_capital,
            'cash': self.current_capital,
            'positions': 0,
            'nav': 1.0
        })
        
        # 遍历数据进行回测
        for i in range(lookback, len(data) - forecast_steps + 1):
            # 获取历史数据
            historical_data = data.iloc[i-lookback:i]
            current_data = data.iloc[i]
            
            # 执行策略
            signal, forecast = strategy_func(historical_data, forecast_steps=forecast_steps)
            
            # 计算当前价格
            current_price = current_data['close']
            
            # 计算波动率（使用历史数据的标准差）
            volatility = historical_data['close'].pct_change().std() if len(historical_data) > 10 else 0.01
            
            # 模拟交易
            trade_result = self.execute_trade(signal, current_price, current_data['date'], volatility)
            
            # 更新净值曲线
            self.update_equity_curve(current_data['date'], current_price)
            
            # 保存交易记录
            if trade_result['action'] != 'hold':
                self.backtest_trades.append(trade_result)
        
        # 计算回测指标
        self.calculate_backtest_metrics()
        
        logger.info("回测完成")
        
        return {
            'results': self.backtest_results,
            'metrics': self.backtest_metrics,
            'trades': self.backtest_trades,
            'positions': self.backtest_positions,
            'equity_curve': self.equity_curve
        }
    
    def execute_trade(self, signal, current_price, date, volatility):
        """执行交易
        
        参数:
        signal: 交易信号 (1=买入, -1=卖出, 0=持有)
        current_price: 当前价格
        date: 交易日期
        volatility: 波动率
        
        返回:
        dict: 交易结果
        """
        # 计算交易成本
        total_transaction_cost = self.transaction_cost + self.commission + self.market_impact
        
        # 模拟滑点
        slippage_amount = current_price * self.slippage
        executed_price = current_price + slippage_amount if signal == 1 else current_price - slippage_amount
        
        trade_result = {
            'date': date,
            'signal': signal,
            'current_price': current_price,
            'executed_price': executed_price,
            'volatility': volatility,
            'transaction_cost': total_transaction_cost,
            'action': 'hold',
            'position_size': 0
        }
        
        # 检查止损条件
        stop_loss_triggered = False
        if len(self.backtest_positions) > 0:
            last_position = self.backtest_positions[-1]
            if last_position['shares'] > 0:
                # 检查是否触发止损
                # 从交易结果中获取止损价格，如果没有则使用默认值
                stop_loss = trade_result.get('stop_loss', executed_price * 0.95)  # 默认5%止损
                take_profit = trade_result.get('take_profit', executed_price * 1.10)  # 默认10%止盈
                
                if current_price <= stop_loss:
                    stop_loss_triggered = True
                    signal = -1  # 强制卖出
                elif current_price >= take_profit:
                    signal = -1  # 强制卖出，锁定利润
        
        # 使用风险管理模块计算仓位，降低仓位大小以控制风险
        position_size = self.risk_manager.calculate_position_size(executed_price, volatility)
        position_size *= 0.3  # 降低仓位至30%，控制风险
        
        # 简化的交易逻辑
        if signal == 1 and position_size > 0:
            # 买入
            shares = position_size
            cost = shares * executed_price * (1 + total_transaction_cost)
            
            if cost <= self.current_capital:
                # 检查是否已有持仓，如果有则不重复买入
                has_position = len(self.backtest_positions) > 0 and self.backtest_positions[-1]['shares'] > 0
                if not has_position:
                    self.current_capital -= cost
                    
                    # 设置更严格的止损止盈
                    sl_tp = self.risk_manager.set_stop_loss_take_profit(executed_price, stop_loss_pct=0.05, take_profit_pct=0.10)
                    
                    trade_result['action'] = 'buy'
                    trade_result['shares'] = shares
                    trade_result['cost'] = cost
                    trade_result['stop_loss'] = sl_tp['stop_loss']
                    trade_result['take_profit'] = sl_tp['take_profit']
                    
                    # 添加仓位记录，包含止损止盈信息
                    self.backtest_positions.append({
                        'date': date,
                        'price': executed_price,
                        'shares': shares,
                        'cash': self.current_capital,
                        'total_value': self.current_capital + shares * executed_price,
                        'stop_loss': sl_tp['stop_loss'],
                        'take_profit': sl_tp['take_profit']
                    })
        elif signal == -1 and len(self.backtest_positions) > 0:
            # 卖出（简化实现，卖出所有持仓）
            last_position = self.backtest_positions[-1]
            shares = last_position['shares']
            
            if shares > 0:
                # 计算卖出收益
                proceeds = shares * executed_price * (1 - total_transaction_cost)
                self.current_capital += proceeds
                
                # 计算交易盈亏
                buy_price = last_position['price']
                profit = proceeds - (shares * buy_price * (1 + total_transaction_cost))
                
                trade_result['action'] = 'sell'
                trade_result['shares'] = shares
                trade_result['proceeds'] = proceeds
                trade_result['profit'] = profit
                trade_result['return_pct'] = profit / (shares * buy_price) * 100
                trade_result['stop_loss_triggered'] = stop_loss_triggered
                
                # 添加仓位记录（清空仓位）
                self.backtest_positions.append({
                    'date': date,
                    'price': executed_price,
                    'shares': 0,
                    'cash': self.current_capital,
                    'total_value': self.current_capital,
                    'stop_loss': 0,
                    'take_profit': 0
                })
        
        return trade_result
    
    def update_equity_curve(self, date, current_price):
        """更新净值曲线
        
        参数:
        date: 日期
        current_price: 当前价格
        """
        # 获取当前持仓
        current_shares = self.backtest_positions[-1]['shares'] if self.backtest_positions else 0
        
        # 计算总权益
        total_equity = self.current_capital + current_shares * current_price
        
        # 计算净值
        nav = total_equity / self.initial_capital
        
        # 更新净值曲线
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'cash': self.current_capital,
            'positions': current_shares * current_price,
            'nav': nav
        })
    
    def calculate_backtest_metrics(self):
        """计算回测指标
        """
        if len(self.equity_curve) < 2:
            return
        
        # 提取净值曲线
        nav_series = pd.DataFrame(self.equity_curve)['nav']
        
        # 计算收益率
        returns = nav_series.pct_change().dropna()
        
        # 保存日收益率序列，用于其他模块使用
        self.daily_returns = returns.values.tolist()
        
        # 使用风险管理模块计算风险指标
        risk_metrics = self.risk_manager.calculate_risk_metrics(returns)
        
        # 计算波动率比较
        if len(returns) > 10:  # 至少10个交易日的数据才能计算有意义的波动率
            actual_returns_np = returns.values
            # 使用历史波动率作为预测波动率的基准
            predicted_volatility = actual_returns_np[:-10].std()  # 使用前90%的数据作为预测基准
            actual_volatility = actual_returns_np[-10:].std()  # 使用最近10天的数据作为实际波动率
            
            # 比较波动率
            volatility_comparison = self.risk_manager.compare_volatility(
                actual_returns=actual_returns_np[-10:],
                predicted_volatility=predicted_volatility
            )
            
            # 将波动率比较结果添加到回测指标中
            self.backtest_metrics['volatility_comparison'] = volatility_comparison
            
            # 输出波动率比较结果
            logger.info(f"波动率比较结果: {volatility_comparison['message']}")
            if volatility_comparison['warning']:
                logger.warning(f"波动率警告: {volatility_comparison['message']}")
        
        # 计算回测专用指标
        backtest_specific_metrics = {
            'total_return': (nav_series.iloc[-1] - nav_series.iloc[0]) / nav_series.iloc[0],
            'annualized_return': (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252 / len(nav_series)) - 1,
            'max_drawdown': self.calculate_max_drawdown(nav_series),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
            'calmar_ratio': risk_metrics.get('calmar_ratio', 0),
            'total_trades': len(self.backtest_trades),
            'avg_trade_duration': self.calculate_avg_trade_duration(),
            'avg_win_size': self.calculate_avg_win_size(),
            'avg_loss_size': self.calculate_avg_loss_size(),
            'best_trade': self.calculate_best_trade(),
            'worst_trade': self.calculate_worst_trade(),
            'profit_to_max_drawdown': risk_metrics.get('total_return', 0) / risk_metrics.get('max_drawdown', 1) if risk_metrics.get('max_drawdown', 0) > 0 else 0
        }
        
        # 合并指标
        self.backtest_metrics = {
            **risk_metrics,
            **backtest_specific_metrics
        }
        
        # 保存回测结果
        self.backtest_results = {
            'start_date': self.equity_curve[0]['date'],
            'end_date': self.equity_curve[-1]['date'],
            'total_days': len(self.equity_curve),
            'final_equity': self.equity_curve[-1]['equity'],
            'final_nav': self.equity_curve[-1]['nav'],
            'total_trades': len(self.backtest_trades),
            'winning_trades': len([t for t in self.backtest_trades if t.get('action') == 'sell' and t.get('profit', 0) > 0]),
            'losing_trades': len([t for t in self.backtest_trades if t.get('action') == 'sell' and t.get('profit', 0) < 0])
        }
        
        logger.info(f"回测指标计算完成，最终净值: {self.equity_curve[-1]['nav']:.4f}")
    
    def calculate_max_drawdown(self, nav_series):
        """计算最大回撤
        
        参数:
        nav_series: 净值序列
        
        返回:
        float: 最大回撤
        """
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        return abs(drawdown.min())
    
    def calculate_win_rate(self):
        """计算胜率
        
        返回:
        float: 胜率
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell']
        if not sell_trades:
            return 0
        
        winning_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
        return winning_trades / len(sell_trades)
    
    def calculate_profit_factor(self):
        """计算盈利因子
        
        返回:
        float: 盈利因子
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell']
        if not sell_trades:
            return 0
        
        total_profit = sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) > 0)
        total_loss = abs(sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0))
        
        if total_loss == 0:
            return float('inf')
        
        return total_profit / total_loss
    
    def calculate_avg_trade_duration(self):
        """计算平均交易 duration
        
        返回:
        float: 平均交易 duration
        """
        # 简化实现，实际应该基于买入和卖出日期计算
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell']
        if not sell_trades:
            return 0
        
        return len(self.equity_curve) / len(sell_trades) if sell_trades else 0
    
    def calculate_avg_win_size(self):
        """计算平均盈利大小
        
        返回:
        float: 平均盈利大小
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell' and t.get('profit', 0) > 0]
        if not sell_trades:
            return 0
        
        return np.mean([t.get('profit', 0) for t in sell_trades])
    
    def calculate_avg_loss_size(self):
        """计算平均亏损大小
        
        返回:
        float: 平均亏损大小
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell' and t.get('profit', 0) < 0]
        if not sell_trades:
            return 0
        
        return abs(np.mean([t.get('profit', 0) for t in sell_trades]))
    
    def calculate_best_trade(self):
        """计算最佳交易
        
        返回:
        float: 最佳交易收益
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell']
        if not sell_trades:
            return 0
        
        return max(t.get('profit', 0) for t in sell_trades)
    
    def calculate_worst_trade(self):
        """计算最差交易
        
        返回:
        float: 最差交易收益
        """
        sell_trades = [t for t in self.backtest_trades if t.get('action') == 'sell']
        if not sell_trades:
            return 0
        
        return min(t.get('profit', 0) for t in sell_trades)
    
    def reset(self):
        """重置回测系统
        """
        self.current_capital = self.initial_capital
        self.backtest_results = {}
        self.backtest_metrics = {}
        self.backtest_trades = []
        self.backtest_positions = []
        self.equity_curve = []
        self.daily_returns = []
        self.risk_manager = RiskManagement(initial_capital=self.initial_capital)
    
    def generate_backtest_report(self):
        """生成回测报告
        
        返回:
        dict: 回测报告
        """
        if not self.backtest_results:
            logger.warning("回测报告生成失败，没有回测结果")
            return {}
        
        report = {
            'summary': {
                'start_date': self.backtest_results['start_date'],
                'end_date': self.backtest_results['end_date'],
                'total_days': self.backtest_results['total_days'],
                'total_trades': self.backtest_results['total_trades'],
                'winning_trades': self.backtest_results['winning_trades'],
                'losing_trades': self.backtest_results['losing_trades'],
                'final_equity': self.backtest_results['final_equity'],
                'final_nav': self.backtest_results['final_nav'],
                'total_return': self.backtest_metrics['total_return']
            },
            'performance_metrics': {
                'annualized_return': self.backtest_metrics['annualized_return'],
                'max_drawdown': self.backtest_metrics['max_drawdown'],
                'sharpe_ratio': self.backtest_metrics['sharpe_ratio'],
                'sortino_ratio': self.backtest_metrics['sortino_ratio'],
                'calmar_ratio': self.backtest_metrics['calmar_ratio'],
                'win_rate': self.backtest_metrics['win_rate'],
                'profit_factor': self.backtest_metrics['profit_factor']
            },
            'risk_metrics': {
                'value_at_risk_95': self.backtest_metrics['value_at_risk_95'],
                'value_at_risk_99': self.backtest_metrics['value_at_risk_99'],
                'conditional_var_95': self.backtest_metrics['conditional_var_95'],
                'annual_std': self.backtest_metrics['annual_std'],
                'profit_to_max_drawdown': self.backtest_metrics['profit_to_max_drawdown']
            },
            'trade_statistics': {
                'avg_trade_duration': self.backtest_metrics['avg_trade_duration'],
                'avg_win_size': self.backtest_metrics['avg_win_size'],
                'avg_loss_size': self.backtest_metrics['avg_loss_size'],
                'best_trade': self.backtest_metrics['best_trade'],
                'worst_trade': self.backtest_metrics['worst_trade']
            },
            'equity_curve': self.equity_curve,
            'trades': self.backtest_trades[:10]  # 只返回前10笔交易
        }
        
        logger.info(f"回测报告生成完成，总收益: {self.backtest_metrics['total_return']:.2%}")
        
        return report
    
    def plot_backtest_results(self):
        """绘制回测结果
        
        返回:
        None
        """
        # 简化实现，实际应该使用matplotlib绘制图表
        logger.info("回测结果可视化（简化版）")
        
        # 打印净值曲线
        print("\n=== 净值曲线 ===")
        for i in range(0, len(self.equity_curve), max(1, len(self.equity_curve) // 10)):
            ec = self.equity_curve[i]
            print(f"{ec['date']}: NAV = {ec['nav']:.4f}, Equity = {ec['equity']:.2f}")
        
        # 打印交易统计
        print("\n=== 交易统计 ===")
        print(f"总交易次数: {len(self.backtest_trades)}")
        print(f"盈利交易: {self.backtest_results['winning_trades']}")
        print(f"亏损交易: {self.backtest_results['losing_trades']}")
        print(f"胜率: {self.backtest_metrics['win_rate']:.2%}")
        print(f"盈利因子: {self.backtest_metrics['profit_factor']:.2f}")
        
        # 打印性能指标
        print("\n=== 性能指标 ===")
        print(f"总收益: {self.backtest_metrics['total_return']:.2%}")
        print(f"年化收益: {self.backtest_metrics['annualized_return']:.2%}")
        print(f"夏普比率: {self.backtest_metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {self.backtest_metrics['max_drawdown']:.2%}")
        print(f"卡玛比率: {self.backtest_metrics['calmar_ratio']:.2f}")
    
    def analyze_model_performance(self):
        """分析模型性能
        
        返回:
        dict: 模型性能分析结果
        """
        if not self.backtest_trades:
            return {}
        
        # 分析模型漂移
        nav_series = pd.DataFrame(self.equity_curve)['nav']
        returns = nav_series.pct_change().dropna()
        
        # 检查模型漂移
        if len(returns) > 100:
            recent_returns = returns.tail(30)
            historical_returns = returns.head(len(returns) - 30)
            model_drift = self.risk_manager.check_model_drift(recent_returns, historical_returns)
        else:
            model_drift = False
        
        # 分析性能衰减
        performance_decay = False
        if len(self.equity_curve) > 60:
            nav_values = [ec['nav'] for ec in self.equity_curve]
            performance_decay = self.risk_manager.monitor_performance_decay(np.array(nav_values))
        
        return {
            'model_drift': model_drift,
            'performance_decay': performance_decay,
            'recent_performance': recent_returns.mean() * 252 if 'recent_returns' in locals() else 0,
            'historical_performance': historical_returns.mean() * 252 if 'historical_returns' in locals() else 0
        }
