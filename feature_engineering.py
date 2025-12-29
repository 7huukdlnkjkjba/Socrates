import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.fund_data = None  # 基金数据
        self.gold9999_data = None  # 黄金9999数据
        self.gold_london_data = None  # 伦敦金现数据
        self.usdcny_data = None  # 美元兑人民币汇率数据
        self.feature_matrix = None  # 特征矩阵
    
    def load_data(self, fund_data, gold9999_data, gold_london_data, usdcny_data):
        """加载所有需要的数据
        
        参数:
        fund_data: 基金数据 (DataFrame)
        gold9999_data: 黄金9999数据 (DataFrame)
        gold_london_data: 伦敦金现数据 (DataFrame)
        usdcny_data: 美元兑人民币汇率数据 (DataFrame)
        """
        # 确保所有数据都有日期索引
        if not isinstance(fund_data.index, pd.DatetimeIndex):
            fund_data.set_index('date', inplace=True)
        if not isinstance(gold9999_data.index, pd.DatetimeIndex):
            gold9999_data.set_index('date', inplace=True)
        if not isinstance(gold_london_data.index, pd.DatetimeIndex):
            gold_london_data.set_index('date', inplace=True)
        if not isinstance(usdcny_data.index, pd.DatetimeIndex):
            usdcny_data.set_index('date', inplace=True)
        
        # 按日期排序
        self.fund_data = fund_data.sort_index()
        self.gold9999_data = gold9999_data.sort_index()
        self.gold_london_data = gold_london_data.sort_index()
        self.usdcny_data = usdcny_data.sort_index()
        
        logger.info("所有数据加载完成")
        logger.info(f"基金数据时间范围: {self.fund_data.index.min()} 到 {self.fund_data.index.max()}")
        logger.info(f"黄金9999数据时间范围: {self.gold9999_data.index.min()} 到 {self.gold9999_data.index.max()}")
        logger.info(f"伦敦金现数据时间范围: {self.gold_london_data.index.min()} 到 {self.gold_london_data.index.max()}")
        logger.info(f"美元兑人民币汇率数据时间范围: {self.usdcny_data.index.min()} 到 {self.usdcny_data.index.max()}")
    
    def align_data(self):
        """对齐所有数据到相同的日期索引"""
        # 找到所有数据的共同日期范围
        common_start = max(
            self.fund_data.index.min(),
            self.gold9999_data.index.min(),
            self.gold_london_data.index.min(),
            self.usdcny_data.index.min()
        )
        
        common_end = min(
            self.fund_data.index.max(),
            self.gold9999_data.index.max(),
            self.gold_london_data.index.max(),
            self.usdcny_data.index.max()
        )
        
        logger.info(f"共同日期范围: {common_start} 到 {common_end}")
        
        # 筛选共同日期范围内的数据
        self.fund_data = self.fund_data.loc[common_start:common_end]
        self.gold9999_data = self.gold9999_data.loc[common_start:common_end]
        self.gold_london_data = self.gold_london_data.loc[common_start:common_end]
        self.usdcny_data = self.usdcny_data.loc[common_start:common_end]
        
        # 进一步确保日期完全对齐（取交集）
        common_dates = self.fund_data.index.intersection(
            self.gold9999_data.index.intersection(
                self.gold_london_data.index.intersection(self.usdcny_data.index)
            )
        )
        
        logger.info(f"对齐后的数据点数量: {len(common_dates)}")
        
        self.fund_data = self.fund_data.loc[common_dates]
        self.gold9999_data = self.gold9999_data.loc[common_dates]
        self.gold_london_data = self.gold_london_data.loc[common_dates]
        self.usdcny_data = self.usdcny_data.loc[common_dates]
    
    def calculate_rolling_correlation(self, window=10):
        """计算基金净值与金价的滚动相关性
        
        参数:
        window: 滚动窗口大小
        """
        logger.info(f"计算滚动相关性，窗口大小: {window}")
        
        # 提取收盘价
        fund_close = self.fund_data['close']
        gold9999_close = self.gold9999_data['close']
        gold_london_close = self.gold_london_data['close']
        
        # 计算滚动相关性
        corr_gold9999 = fund_close.rolling(window=window).corr(gold9999_close)
        corr_gold_london = fund_close.rolling(window=window).corr(gold_london_close)
        
        # 将相关性添加到特征矩阵
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=fund_close.index)
        
        self.feature_matrix[f'corr_fund_gold9999_{window}d'] = corr_gold9999
        self.feature_matrix[f'corr_fund_gold_london_{window}d'] = corr_gold_london
    
    def calculate_price_spread(self):
        """计算不同金价之间的价差"""
        logger.info("计算金价价差")
        
        # 计算黄金9999与伦敦金现的价差（需要考虑单位转换）
        # 伦敦金现单位是美元/盎司，黄金9999单位是人民币/克
        # 1盎司 ≈ 31.1035克
        oz_to_gram = 31.1035
        
        # 计算人民币计价的伦敦金现价格
        gold_london_cny = self.gold_london_data['close'] * self.usdcny_data['close'] / oz_to_gram
        
        # 计算价差
        spread = self.gold9999_data['close'] - gold_london_cny
        
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=spread.index)
        
        self.feature_matrix['gold_price_spread'] = spread
        self.feature_matrix['gold_london_cny'] = gold_london_cny  # 人民币计价的伦敦金现价格
    
    def calculate_theoretical_nav(self):
        """计算汇率折算后的理论净值
        
        理论净值 = 基金实际净值 * (基准汇率 / 当前汇率)
        用于衡量汇率变动对基金净值的影响
        """
        logger.info("计算理论净值")
        
        # 使用第一个可用日期的汇率作为基准汇率
        base_exchange_rate = self.usdcny_data['close'].iloc[0]
        
        # 计算理论净值
        theoretical_nav = self.fund_data['close'] * (base_exchange_rate / self.usdcny_data['close'])
        
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=theoretical_nav.index)
        
        self.feature_matrix['theoretical_nav'] = theoretical_nav
        self.feature_matrix['nav_deviation'] = self.fund_data['close'] - theoretical_nav  # 实际净值与理论净值的偏差
    
    def calculate_momentum_features(self, windows=[5, 10, 20]):
        """计算动量特征
        
        参数:
        windows: 计算动量的窗口大小列表
        """
        logger.info(f"计算动量特征，窗口大小: {windows}")
        
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=self.fund_data.index)
        
        # 基金净值动量
        for window in windows:
            # 收益率动量
            self.feature_matrix[f'fund_momentum_{window}d'] = self.fund_data['close'].pct_change(periods=window)
            # 价格变化动量
            self.feature_matrix[f'fund_price_change_{window}d'] = self.fund_data['close'] - self.fund_data['close'].shift(window)
        
        # 黄金价格动量
        for window in windows:
            self.feature_matrix[f'gold9999_momentum_{window}d'] = self.gold9999_data['close'].pct_change(periods=window)
            self.feature_matrix[f'gold_london_momentum_{window}d'] = self.gold_london_data['close'].pct_change(periods=window)
    
    def calculate_volatility_features(self, windows=[10, 20, 30]):
        """计算波动率特征
        
        参数:
        windows: 计算波动率的窗口大小列表
        """
        logger.info(f"计算波动率特征，窗口大小: {windows}")
        
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=self.fund_data.index)
        
        # 计算对数收益率
        fund_log_returns = np.log(self.fund_data['close'] / self.fund_data['close'].shift(1))
        gold9999_log_returns = np.log(self.gold9999_data['close'] / self.gold9999_data['close'].shift(1))
        gold_london_log_returns = np.log(self.gold_london_data['close'] / self.gold_london_data['close'].shift(1))
        
        # 计算滚动波动率（年化）
        annualization_factor = np.sqrt(252)  # 年化因子
        
        for window in windows:
            self.feature_matrix[f'fund_volatility_{window}d'] = fund_log_returns.rolling(window=window).std() * annualization_factor
            self.feature_matrix[f'gold9999_volatility_{window}d'] = gold9999_log_returns.rolling(window=window).std() * annualization_factor
            self.feature_matrix[f'gold_london_volatility_{window}d'] = gold_london_log_returns.rolling(window=window).std() * annualization_factor
    
    def calculate_technical_indicators(self):
        """计算技术指标特征"""
        logger.info("计算技术指标特征")
        
        if self.feature_matrix is None:
            self.feature_matrix = pd.DataFrame(index=self.fund_data.index)
        
        # 移动平均线 (MA)
        for window in [5, 10, 20, 30]:
            self.feature_matrix[f'fund_ma_{window}d'] = self.fund_data['close'].rolling(window=window).mean()
            self.feature_matrix[f'gold9999_ma_{window}d'] = self.gold9999_data['close'].rolling(window=window).mean()
        
        # 相对强弱指标 (RSI)
        for window in [14, 21]:
            # 基金RSI
            delta = self.fund_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            self.feature_matrix[f'fund_rsi_{window}d'] = 100 - (100 / (1 + rs))
            
            # 黄金RSI
            delta_gold = self.gold9999_data['close'].diff()
            gain_gold = (delta_gold.where(delta_gold > 0, 0)).rolling(window=window).mean()
            loss_gold = (-delta_gold.where(delta_gold < 0, 0)).rolling(window=window).mean()
            rs_gold = gain_gold / loss_gold
            self.feature_matrix[f'gold9999_rsi_{window}d'] = 100 - (100 / (1 + rs_gold))
        
        # 布林带
        for window in [20]:
            # 基金布林带
            ma = self.fund_data['close'].rolling(window=window).mean()
            std = self.fund_data['close'].rolling(window=window).std()
            self.feature_matrix[f'fund_bollinger_mid_{window}d'] = ma
            self.feature_matrix[f'fund_bollinger_upper_{window}d'] = ma + 2 * std
            self.feature_matrix[f'fund_bollinger_lower_{window}d'] = ma - 2 * std
            self.feature_matrix[f'fund_bollinger_width_{window}d'] = (self.feature_matrix[f'fund_bollinger_upper_{window}d'] - self.feature_matrix[f'fund_bollinger_lower_{window}d']) / ma
    
    def create_feature_matrix(self, rolling_correlation_windows=[10, 20], momentum_windows=[5, 10, 20], volatility_windows=[10, 20, 30]):
        """创建完整的特征矩阵
        
        参数:
        rolling_correlation_windows: 滚动相关性的窗口大小列表
        momentum_windows: 动量特征的窗口大小列表
        volatility_windows: 波动率特征的窗口大小列表
        """
        logger.info("开始创建特征矩阵")
        
        # 首先对齐所有数据
        self.align_data()
        
        # 初始化特征矩阵
        self.feature_matrix = pd.DataFrame(index=self.fund_data.index)
        
        # 计算各种特征
        for window in rolling_correlation_windows:
            self.calculate_rolling_correlation(window=window)
        
        self.calculate_price_spread()
        self.calculate_theoretical_nav()
        self.calculate_momentum_features(windows=momentum_windows)
        self.calculate_volatility_features(windows=volatility_windows)
        self.calculate_technical_indicators()
        
        # 添加原始数据作为特征
        self.feature_matrix['fund_close'] = self.fund_data['close']
        self.feature_matrix['gold9999_close'] = self.gold9999_data['close']
        self.feature_matrix['gold_london_close'] = self.gold_london_data['close']
        self.feature_matrix['usdcny_close'] = self.usdcny_data['close']
        
        # 添加时间特征
        self.feature_matrix['day_of_week'] = self.feature_matrix.index.dayofweek
        self.feature_matrix['month'] = self.feature_matrix.index.month
        self.feature_matrix['quarter'] = self.feature_matrix.index.quarter
        
        # 处理无穷大值和非常大的值
        self.feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 删除包含缺失值的行
        self.feature_matrix.dropna(inplace=True)
        
        logger.info(f"特征矩阵创建完成，形状: {self.feature_matrix.shape}")
        logger.info(f"特征数量: {len(self.feature_matrix.columns)}")
        
        return self.feature_matrix
    
    def get_feature_importance(self, target_column='fund_close', n_estimators=100, random_state=42):
        """使用随机森林计算特征重要性
        
        参数:
        target_column: 目标列名称
        n_estimators: 随机森林估计器数量
        random_state: 随机种子
        """
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info("计算特征重要性")
        
        if self.feature_matrix is None:
            raise ValueError("请先创建特征矩阵")
        
        # 准备数据
        X = self.feature_matrix.drop(columns=[target_column])
        y = self.feature_matrix[target_column]
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("特征重要性计算完成")
        
        return feature_importance
    
    def visualize_feature_correlation(self, figsize=(15, 12)):
        """可视化特征相关性
        
        参数:
        figsize: 图表大小
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.feature_matrix is None:
            raise ValueError("请先创建特征矩阵")
        
        logger.info("可视化特征相关性")
        
        # 计算相关矩阵
        corr_matrix = self.feature_matrix.corr()
        
        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True, linewidths=0.5)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.show()
    
    def get_feature_matrix(self):
        """获取特征矩阵"""
        return self.feature_matrix
    
    def save_feature_matrix(self, file_path):
        """保存特征矩阵到文件
        
        参数:
        file_path: 文件路径
        """
        if self.feature_matrix is None:
            raise ValueError("请先创建特征矩阵")
        
        self.feature_matrix.to_csv(file_path)
        logger.info(f"特征矩阵已保存到: {file_path}")
    
    def load_feature_matrix(self, file_path):
        """从文件加载特征矩阵
        
        参数:
        file_path: 文件路径
        """
        self.feature_matrix = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"特征矩阵已从 {file_path} 加载，形状: {self.feature_matrix.shape}")
