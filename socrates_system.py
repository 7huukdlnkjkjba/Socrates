import pandas as pd
import numpy as np
import logging
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from prediction import TimeSeriesPredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocratesSystem:
    def __init__(self):
        self.data_processor = None
        self.feature_engineer = None
        self.predictor = None
        self.fund_data = None
        self.gold9999_data = None
        self.gold_london_data = None
        self.usdcny_data = None
        self.feature_matrix = None
        self.predictions = None
        self.evaluations = None
        self.best_model = None
    
    def crawl_data(self):
        """爬取多源数据"""
        logger.info("=== 开始爬取多源数据 ===")
        
        # 创建DataProcessor实例
        self.data_processor = DataProcessor()
        
        # 爬取各种数据源
        logger.info("爬取上海黄金交易所Au9999数据...")
        self.gold9999_data = self.data_processor.crawl_gold9999()
        
        logger.info("爬取伦敦金数据...")
        self.gold_london_data = self.data_processor.crawl_gold_london()
        
        logger.info("爬取美元兑人民币汇率数据...")
        self.usdcny_data = self.data_processor.crawl_usdcny()
        
        logger.info("=== 数据爬取完成 ===")
        
    def generate_simulated_fund_data(self):
        """生成模拟的基金数据（当实际爬取受限或需要测试时使用）"""
        logger.info("=== 生成模拟基金数据 ===")
        
        # 使用与其他数据源相同的日期范围
        dates = self.gold9999_data['date']
        
        # 创建模拟的基金数据
        np.random.seed(42)
        
        # 生成模拟的基金净值（与黄金价格有一定相关性）
        gold9999_close = self.gold9999_data['close'].values
        # 添加一些噪声，使得基金净值与黄金价格有一定相关性但不完全相同
        fund_close = 1.5 + 0.001 * gold9999_close + np.random.normal(0, 0.02, len(gold9999_close))
        
        # 创建基金数据DataFrame
        self.fund_data = pd.DataFrame({
            'date': dates,
            'open': fund_close * np.random.uniform(0.995, 1.005, len(fund_close)),
            'high': fund_close * np.random.uniform(1.0, 1.01, len(fund_close)),
            'low': fund_close * np.random.uniform(0.99, 1.0, len(fund_close)),
            'close': fund_close,
            'volume': np.random.randint(100000, 1000000, len(fund_close))
        })
        
        logger.info(f"生成的基金数据：{len(self.fund_data)}条记录")
        logger.info(f"数据日期范围: {self.fund_data['date'].min()} 到 {self.fund_data['date'].max()}")
        logger.info("=== 模拟基金数据生成完成 ===")
    
    def feature_engineering(self):
        """进行特征工程"""
        logger.info("=== 开始特征工程 ===")
        
        # 创建FeatureEngineer实例
        self.feature_engineer = FeatureEngineer()
        
        # 加载数据
        logger.info("加载数据到特征工程模块...")
        self.feature_engineer.load_data(self.fund_data, self.gold9999_data, self.gold_london_data, self.usdcny_data)
        
        # 创建特征矩阵
        logger.info("创建特征矩阵...")
        self.feature_matrix = self.feature_engineer.create_feature_matrix(
            rolling_correlation_windows=[10, 20],
            momentum_windows=[5, 10, 20],
            volatility_windows=[10, 20, 30]
        )
        
        logger.info(f"特征矩阵创建成功，形状: {self.feature_matrix.shape}")
        logger.info(f"特征数量: {len(self.feature_matrix.columns)}")
        logger.info(f"特征矩阵日期范围: {self.feature_matrix.index.min()} 到 {self.feature_matrix.index.max()}")
        
        # 计算特征重要性
        logger.info("计算特征重要性...")
        feature_importance = self.feature_engineer.get_feature_importance(target_column='fund_close')
        logger.info("特征重要性前10名:")
        logger.info(feature_importance.head(10).to_string())
        
        logger.info("=== 特征工程完成 ===")
    
    def train_predictors(self):
        """训练预测模型"""
        logger.info("=== 开始训练预测模型 ===")
        
        # 创建TimeSeriesPredictor实例
        self.predictor = TimeSeriesPredictor(self.feature_matrix, target_column='fund_close')
        
        # 训练各种模型
        logger.info("训练移动平均(MA)模型...")
        self.predictor.moving_average(window_size=3, forecast_steps=1)
        
        logger.info("训练预测移动平均(PMA)模型...")
        self.predictor.predicted_moving_average(window_size=3, lookback_period=10, forecast_steps=1)
        
        logger.info("训练指数平滑(ES)模型...")
        self.predictor.exponential_smoothing(alpha=0.2, forecast_steps=1)
        
        logger.info("训练XGBoost模型...")
        self.predictor.xgboost(feature_matrix=self.feature_matrix, forecast_steps=1)
        
        logger.info("训练LSTM模型...")
        self.predictor.lstm(lookback=10, forecast_steps=1, feature_matrix=self.feature_matrix)
        
        logger.info("=== 模型训练完成 ===")
    
    def evaluate_models(self):
        """评估模型性能"""
        logger.info("=== 开始评估模型性能 ===")
        
        self.evaluations = self.predictor.evaluate(feature_matrix=self.feature_matrix)
        
        logger.info("模型评估结果:")
        for model_name, metrics in self.evaluations.items():
            logger.info(f"{model_name}: RMSE = {metrics['RMSE']:.6f}, MAE = {metrics['MAE']:.6f}")
        
        # 选择最优模型
        self.best_model = self.predictor.get_best_model()
        logger.info(f"最优模型: {self.best_model} (RMSE = {self.evaluations[self.best_model]['RMSE']:.6f})")
        
        logger.info("=== 模型评估完成 ===")
    
    def make_predictions(self, forecast_steps=5):
        """使用XGBoost模型进行预测（强制使用，因为它能生成动态预测结果）"""
        logger.info(f"=== 强制使用XGBoost模型预测未来{forecast_steps}天 ===")
        
        # 直接使用XGBoost进行预测，而不是最优模型
        self.predictions = self.predictor.xgboost(feature_matrix=self.feature_matrix, forecast_steps=forecast_steps)
        
        logger.info(f"未来{forecast_steps}天预测结果:")
        for i, pred in enumerate(self.predictions):
            logger.info(f"第{i+1}天: {pred:.6f}")
        
        logger.info("=== 预测完成 ===")
        
        return self.predictions
    
    def run_pipeline(self, forecast_steps=5, use_simulated_fund=True):
        """运行完整的预测流程
        
        参数:
        forecast_steps: 预测步数
        use_simulated_fund: 是否使用模拟基金数据
        """
        logger.info("=== 苏格拉底时间序列预测系统开始运行 ===")
        
        # 爬取多源数据
        self.crawl_data()
        
        # 如果需要，生成模拟基金数据
        if use_simulated_fund:
            self.generate_simulated_fund_data()
        
        # 进行特征工程
        self.feature_engineering()
        
        # 训练预测模型
        self.train_predictors()
        
        # 评估模型性能
        self.evaluate_models()
        
        # 进行预测
        predictions = self.make_predictions(forecast_steps=forecast_steps)
        
        logger.info("=== 苏格拉底时间序列预测系统运行完成 ===")
        
        return {
            'predictions': predictions,
            'best_model': self.best_model,
            'evaluations': self.evaluations,
            'feature_matrix': self.feature_matrix,
            'fund_data': self.fund_data,
            'gold9999_data': self.gold9999_data,
            'gold_london_data': self.gold_london_data,
            'usdcny_data': self.usdcny_data
        }

if __name__ == "__main__":
    # 创建系统实例
    socrates = SocratesSystem()
    
    # 运行完整流程，预测未来5天
    result = socrates.run_pipeline(forecast_steps=5, use_simulated_fund=True)
