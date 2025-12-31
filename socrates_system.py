import pandas as pd
import numpy as np
import logging
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from prediction import TimeSeriesPredictor
from pytorch_predictor import PyTorchTimeSeriesPredictor
from traditional_forecasting import TraditionalForecasting

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocratesSystem:
    def __init__(self):
        self.data_processor = None
        self.feature_engineer = None
        self.predictor = None
        self.pytorch_predictor = None
        self.fund_data = None
        self.gold9999_data = None
        self.gold_london_data = None
        self.usdcny_data = None
        self.feature_matrix = None
        self.predictions = None
        self.evaluations = None
        self.best_model = None
        self.pytorch_predictions = None
        self.ensemble_predictions = None
        
        # 验证相关属性
        self.validation_history = []  # 验证历史记录
        self.prediction_history = []   # 预测历史记录
        self.actual_results = {}       # 实际结果历史记录
        self.model_validation_metrics = {}  # 模型验证指标
        self.validation_weight_adjustments = []  # 基于验证的权重调整历史
        self.prediction_id_counter = 0  # 预测ID计数器，用于唯一标识每个预测
        
        # 传统预测方法相关
        self.traditional_forecaster = None
        self.traditional_predictions = None
        
        # 添加系统配置
        self.use_real_data = True  # 默认使用真实数据
        self.force_refresh_cache = False  # 强制刷新缓存
        self.skip_user_prompts = False  # 跳过用户提示
        
        # PyTorch模型配置
        self.pytorch_config = {
            'lstm': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
            'gru': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
            'tft': {'hidden_size': 64, 'num_heads': 4, 'num_layers': 2},
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'gradient_clip_val': 1.0,  # 添加梯度裁剪
                'patience': 10  # 早停耐心值
            }
        }
    
    def crawl_data(self):
        """爬取多源数据，支持强制刷新缓存"""
        logger.info("=== 开始爬取多源数据 ===")
        
        # 创建DataProcessor实例
        self.data_processor = DataProcessor()
        
        # 设置是否强制刷新缓存
        if self.force_refresh_cache:
            logger.info("强制刷新缓存模式开启")
        
        # 爬取各种数据源
        logger.info("爬取上海黄金交易所Au9999数据...")
        self.gold9999_data = self.data_processor.crawl_gold9999()
        
        logger.info("爬取伦敦金数据...")
        self.gold_london_data = self.data_processor.crawl_gold_london()
        
        logger.info("爬取美元兑人民币汇率数据...")
        self.usdcny_data = self.data_processor.crawl_usdcny()
        
        logger.info("爬取博时黄金C(002611)数据...")
        self.fund_data = self.data_processor.crawl_boshi_gold_c()
        
        # 检查数据是否有效
        if self.fund_data is None or len(self.fund_data) == 0:
            logger.error("基金数据获取失败，无法继续")
            raise Exception("无法获取博时黄金C(002611)的真实数据")
        else:
            logger.info(f"成功获取基金数据: {len(self.fund_data)}条记录")
            logger.info(f"基金数据日期范围: {self.fund_data['date'].min()} 到 {self.fund_data['date'].max()}")
        
        logger.info("=== 数据爬取完成 ===")
    
    def generate_simulated_fund_data(self):
        """生成模拟的基金数据（当实际爬取受限或需要测试时使用）"""
        logger.info("=== 生成模拟基金数据 ===")
        
        # 检查是否有黄金数据作为基础
        if self.gold9999_data is None:
            logger.error("没有黄金数据作为模拟基础")
            return
        
        # 使用与其他数据源相同的日期范围
        dates = self.gold9999_data['date']
        
        # 创建模拟的基金数据
        np.random.seed(42)
        
        # 生成模拟的基金净值（与黄金价格有一定相关性）
        gold9999_close = self.gold9999_data['close'].values
        
        # 确保价格在合理范围内
        if len(gold9999_close) > 0:
            gold_mean = np.mean(gold9999_close)
            gold_std = np.std(gold9999_close)
            
            # 生成更合理的模拟数据
            base_price = 1.5  # 基础价格
            correlation_factor = 0.001  # 与黄金的相关性因子
            noise_scale = 0.02  # 噪声尺度
            
            # 使用黄金价格创建趋势
            gold_normalized = (gold9999_close - gold_mean) / gold_std if gold_std > 0 else 0
            fund_close = base_price + correlation_factor * gold_normalized * gold_std + np.random.normal(0, noise_scale, len(gold9999_close))
            
            # 确保价格为正值
            fund_close = np.maximum(fund_close, 0.1)
        else:
            # 如果没有黄金数据，生成简单的随机数据
            fund_close = 1.5 + np.random.normal(0, 0.1, len(dates))
            fund_close = np.maximum(fund_close, 0.1)
        
        # 创建基金数据DataFrame
        self.fund_data = pd.DataFrame({
            'date': dates,
            'open': fund_close * np.random.uniform(0.995, 1.005, len(fund_close)),
            'high': fund_close * np.random.uniform(1.0, 1.01, len(fund_close)),
            'low': fund_close * np.random.uniform(0.99, 1.0, len(fund_close)),
            'close': fund_close,
            'volume': np.random.randint(100000, 1000000, len(fund_close))
        })
        
        logger.warning(f"生成的基金数据：{len(self.fund_data)}条记录")
        logger.warning(f"数据日期范围: {self.fund_data['date'].min()} 到 {self.fund_data['date'].max()}")
        logger.warning("=== 模拟基金数据生成完成 ===")
        logger.warning("注意：本次预测使用的是模拟数据，而非真实市场数据！")
    
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
        
        # 检查特征矩阵是否有效
        if self.feature_matrix is None or len(self.feature_matrix) == 0:
            logger.error("特征矩阵创建失败！")
            return False
        
        logger.info(f"特征矩阵创建成功，形状: {self.feature_matrix.shape}")
        logger.info(f"特征数量: {len(self.feature_matrix.columns)}")
        logger.info(f"特征矩阵日期范围: {self.feature_matrix.index.min()} 到 {self.feature_matrix.index.max()}")
        
        # 检查特征中是否有异常值
        fund_close_series = self.feature_matrix['fund_close']
        fund_mean = fund_close_series.mean()
        fund_std = fund_close_series.std()
        
        logger.info(f"基金价格统计: 均值={fund_mean:.4f}, 标准差={fund_std:.4f}")
        logger.info(f"基金价格范围: [{fund_close_series.min():.4f}, {fund_close_series.max():.4f}]")
        
        # 计算特征重要性
        logger.info("计算特征重要性...")
        try:
            feature_importance = self.feature_engineer.get_feature_importance(target_column='fund_close')
            if feature_importance is not None and len(feature_importance) > 0:
                logger.info("特征重要性前10名:")
                logger.info(feature_importance.head(10).to_string())
            else:
                logger.warning("特征重要性计算失败或结果为空")
        except Exception as e:
            logger.warning(f"计算特征重要性时出错: {e}")
        
        logger.info("=== 特征工程完成 ===")
        return True
    
    def train_predictors(self):
        """训练预测模型，修复PyTorch模型训练问题"""
        logger.info("=== 开始训练预测模型 ===")
        
        # 创建传统预测器实例
        self.predictor = TimeSeriesPredictor(self.feature_matrix, target_column='fund_close')
        
        # 训练各种传统模型
        logger.info("训练移动平均(MA)模型...")
        self.predictor.moving_average(window_size=3, forecast_steps=1)
        
        logger.info("训练预测移动平均(PMA)模型...")
        self.predictor.predicted_moving_average(window_size=3, lookback_period=10, forecast_steps=1)
        
        logger.info("训练指数平滑(ES)模型...")
        self.predictor.exponential_smoothing(alpha=0.2, forecast_steps=1)
        
        logger.info("训练XGBoost模型...")
        self.predictor.xgboost(feature_matrix=self.feature_matrix, forecast_steps=1)
        
        logger.info("训练传统LSTM模型...")
        self.predictor.lstm(lookback=10, forecast_steps=1, feature_matrix=self.feature_matrix)
        
        # 创建PyTorch预测器实例并训练高级模型
        logger.info("=== 开始训练PyTorch高级模型 ===")
        self.pytorch_predictor = PyTorchTimeSeriesPredictor(self.feature_matrix, target_column='fund_close')
        
        # 配置PyTorch训练参数
        config = self.pytorch_config
        train_params = config['training']
        
        logger.info("训练PyTorch LSTM模型...")
        self.pytorch_predictor.lstm_pytorch(
            lookback=20, 
            forecast_steps=1, 
            hidden_size=config['lstm']['hidden_size'], 
            num_layers=config['lstm']['num_layers'],
            epochs=train_params['epochs'], 
            batch_size=train_params['batch_size'], 
            learning_rate=train_params['learning_rate'],
            feature_matrix=self.feature_matrix
        )
        
        logger.info("训练PyTorch GRU模型...")
        self.pytorch_predictor.gru_pytorch(
            lookback=20, 
            forecast_steps=1, 
            hidden_size=config['gru']['hidden_size'], 
            num_layers=config['gru']['num_layers'],
            epochs=train_params['epochs'], 
            batch_size=train_params['batch_size'], 
            learning_rate=train_params['learning_rate'],
            feature_matrix=self.feature_matrix
        )
        
        # 暂时禁用Transformer，需要修复
        # logger.info("训练PyTorch Transformer模型...")
        # self.pytorch_predictor.transformer_pytorch(
        #     lookback=20, 
        #     forecast_steps=1, 
        #     d_model=64, 
        #     nhead=4, 
        #     epochs=30,  # 减少epochs，Transformer更容易过拟合
        #     batch_size=32, 
        #     learning_rate=0.0005,  # 更小的学习率
        #     patience=10,
        #     feature_matrix=self.feature_matrix
        # )
        
        logger.info("训练PyTorch TFT模型...")
        self.pytorch_predictor.tft_pytorch(
            lookback=20, 
            forecast_steps=1, 
            hidden_size=config['tft']['hidden_size'], 
            num_heads=config['tft']['num_heads'], 
            num_layers=config['tft']['num_layers'],
            epochs=train_params['epochs'], 
            batch_size=train_params['batch_size'], 
            learning_rate=train_params['learning_rate'],
            feature_matrix=self.feature_matrix
        )
        
        logger.info("=== 所有模型训练完成 ===")
    
    def evaluate_models(self):
        """评估模型性能"""
        logger.info("=== 开始评估模型性能 ===")
        
        try:
            self.evaluations = self.predictor.evaluate(feature_matrix=self.feature_matrix)
            
            if self.evaluations:
                logger.info("模型评估结果:")
                for model_name, metrics in self.evaluations.items():
                    logger.info(f"{model_name}: RMSE = {metrics['RMSE']:.6f}, MAE = {metrics['MAE']:.6f}")
                
                # 选择最优模型
                self.best_model = self.predictor.get_best_model()
                if self.best_model:
                    logger.info(f"最优模型: {self.best_model} (RMSE = {self.evaluations[self.best_model]['RMSE']:.6f})")
                else:
                    logger.warning("无法确定最优模型")
            else:
                logger.warning("模型评估结果为空")
                
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            self.evaluations = {}
            self.best_model = None
        
        logger.info("=== 模型评估完成 ===")
    
    def calculate_dynamic_weights(self, test_size=0.2, lookback=20):
        """根据模型最近的预测性能计算动态权重，修复权重计算问题"""
        logging.info("计算动态权重...")
        
        # 创建测试数据
        data = self.feature_matrix.values
        target_column = self.feature_matrix.columns.get_loc('fund_close')
        
        # 划分训练和测试数据
        test_start_idx = int(len(data) * (1 - test_size))
        test_data = data[test_start_idx:]
        
        if len(test_data) < 5:
            logging.warning("测试数据太少，使用默认权重")
            return self.get_default_weights()
        
        # 计算每个模型的预测误差
        model_errors = {}
        predictions = {}
        
        # 获取当前基金价格范围用于归一化
        fund_close_values = self.feature_matrix['fund_close'].values
        fund_mean = np.mean(fund_close_values)
        fund_std = np.std(fund_close_values)
        
        # 测试XGBoost
        try:
            pred_steps = min(5, len(test_data))
            xgboost_preds = self.predictor.xgboost(feature_matrix=self.feature_matrix, forecast_steps=pred_steps)
            
            if xgboost_preds is not None and len(xgboost_preds) > 0:
                # 计算误差（使用最后几个实际值与预测值比较）
                actual_values = self.feature_matrix['fund_close'].iloc[-len(xgboost_preds):].values
                
                # 归一化误差
                normalized_actual = (actual_values - fund_mean) / (fund_std + 1e-8)
                normalized_preds = (xgboost_preds - fund_mean) / (fund_std + 1e-8)
                xgboost_error = np.mean(np.abs(normalized_actual - normalized_preds))
                
                model_errors['xgboost'] = xgboost_error
                predictions['xgboost'] = xgboost_preds
            else:
                model_errors['xgboost'] = float('inf')
        except Exception as e:
            logging.warning(f"XGBoost评估失败: {e}")
            model_errors['xgboost'] = float('inf')
        
        # 测试PyTorch模型
        for model_name, model_func in {
            'lstm_py': lambda steps: self.pytorch_predictor.lstm_pytorch(
                lookback=lookback, forecast_steps=steps, 
                hidden_size=64, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            ),
            'gru_py': lambda steps: self.pytorch_predictor.gru_pytorch(
                lookback=lookback, forecast_steps=steps, 
                hidden_size=64, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            ),
            'tft_py': lambda steps: self.pytorch_predictor.tft_pytorch(
                lookback=lookback, forecast_steps=steps, 
                hidden_size=64, num_heads=4, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            )
        }.items():
            try:
                pred_steps = min(5, len(test_data))
                preds = model_func(pred_steps)
                
                if preds is not None and len(preds) > 0:
                    actual_values = self.feature_matrix['fund_close'].iloc[-len(preds):].values
                    
                    # 归一化误差
                    normalized_actual = (actual_values - fund_mean) / (fund_std + 1e-8)
                    normalized_preds = (preds - fund_mean) / (fund_std + 1e-8)
                    error = np.mean(np.abs(normalized_actual - normalized_preds))
                    
                    # 检查预测是否合理（防止梯度爆炸）
                    if np.any(np.abs(preds) > 100 * fund_mean):
                        logging.warning(f"{model_name} 预测值异常，给予较高误差")
                        error = 10.0  # 给予较高误差
                    
                    model_errors[model_name] = error
                    predictions[model_name] = preds
                else:
                    model_errors[model_name] = float('inf')
            except Exception as e:
                logging.warning(f"{model_name}评估失败: {e}")
                model_errors[model_name] = float('inf')
        
        # 计算权重：误差越小，权重越大（使用倒数变换）
        weights = {}
        valid_errors = {k: v for k, v in model_errors.items() if v != float('inf') and v > 0}
        
        if len(valid_errors) > 0:
            # 使用softmax的变体计算权重
            min_error = min(valid_errors.values())
            adjusted_errors = {k: np.exp(-v / (min_error + 1e-8)) for k, v in valid_errors.items()}
            total_score = sum(adjusted_errors.values())
            
            if total_score > 0:
                for model, score in adjusted_errors.items():
                    weights[model] = score / total_score
            else:
                weights = self.get_default_weights()
        else:
            weights = self.get_default_weights()
        
        logging.info(f"动态权重计算完成: {weights}")
        
        return weights
    
    def get_default_weights(self):
        """获取默认模型权重"""
        return {
            'xgboost': 0.40,
            'lstm_py': 0.20,
            'gru_py': 0.15,
            'tft_py': 0.15,
            'i_ching': 0.03,
            'liu_yao': 0.03,
            'qimen': 0.02,
            'tarot': 0.02
        }
    
    def make_predictions(self, forecast_steps=5):
        """实现多模型集成策略，修复预测值异常问题"""
        logger.info(f"=== 开始多模型集成预测未来{forecast_steps}天 ===")
        
        # 获取各个模型的预测结果
        model_predictions = {}
        
        logger.info("获取XGBoost模型预测结果...")
        try:
            xgboost_preds = self.predictor.xgboost(feature_matrix=self.feature_matrix, forecast_steps=forecast_steps)
            model_predictions['xgboost'] = xgboost_preds
        except Exception as e:
            logger.error(f"XGBoost预测失败: {e}")
            model_predictions['xgboost'] = [self.feature_matrix['fund_close'].iloc[-1]] * forecast_steps
        
        logger.info("获取PyTorch LSTM模型预测结果...")
        try:
            lstm_py_preds = self.pytorch_predictor.lstm_pytorch(
                lookback=20, forecast_steps=forecast_steps, 
                hidden_size=64, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            )
            model_predictions['lstm_py'] = lstm_py_preds
        except Exception as e:
            logger.error(f"LSTM预测失败: {e}")
            model_predictions['lstm_py'] = [self.feature_matrix['fund_close'].iloc[-1]] * forecast_steps
        
        logger.info("获取PyTorch GRU模型预测结果...")
        try:
            gru_py_preds = self.pytorch_predictor.gru_pytorch(
                lookback=20, forecast_steps=forecast_steps, 
                hidden_size=64, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            )
            model_predictions['gru_py'] = gru_py_preds
        except Exception as e:
            logger.error(f"GRU预测失败: {e}")
            model_predictions['gru_py'] = [self.feature_matrix['fund_close'].iloc[-1]] * forecast_steps
        
        # Transformer暂时禁用
        logger.info("PyTorch Transformer模型暂时禁用...")
        transformer_py_preds = [self.feature_matrix['fund_close'].iloc[-1]] * forecast_steps
        model_predictions['transformer_py'] = transformer_py_preds
        
        logger.info("获取PyTorch TFT模型预测结果...")
        try:
            tft_py_preds = self.pytorch_predictor.tft_pytorch(
                lookback=20, forecast_steps=forecast_steps, 
                hidden_size=64, num_heads=4, num_layers=2, epochs=30, 
                batch_size=32, feature_matrix=self.feature_matrix
            )
            model_predictions['tft_py'] = tft_py_preds
        except Exception as e:
            logger.error(f"TFT预测失败: {e}")
            model_predictions['tft_py'] = [self.feature_matrix['fund_close'].iloc[-1]] * forecast_steps
        
        # 获取传统预测方法的结果
        logger.info("获取传统预测方法结果...")
        current_price = self.feature_matrix['fund_close'].iloc[-1]
        self.traditional_forecaster = TraditionalForecasting(current_price)
        traditional_results = self.traditional_forecaster.get_all_traditional_predictions()
        
        # 为每个传统方法创建预测序列
        for method_name in ['i_ching', 'liu_yao', 'qimen', 'tarot']:
            if method_name in traditional_results:
                predicted_price = traditional_results[method_name]['predicted_price']
                model_predictions[method_name] = [predicted_price] * forecast_steps
        
        # 保存传统预测结果
        self.traditional_predictions = traditional_results
        
        # 实现集成策略（动态加权平均，基于模型最近性能）
        logger.info("进行多模型集成...")
        
        # 计算动态权重
        weights = self.calculate_dynamic_weights()
        
        # 添加传统方法的权重
        for method_name in ['i_ching', 'liu_yao', 'qimen', 'tarot']:
            if method_name not in weights:
                weights[method_name] = 0.03  # 默认小权重
        
        # 归一化权重确保总和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 验证和清理预测值
        current_price = self.feature_matrix['fund_close'].iloc[-1]
        for model_name, preds in model_predictions.items():
            if preds is None or len(preds) == 0:
                model_predictions[model_name] = [current_price] * forecast_steps
            else:
                # 检查预测值是否合理
                cleaned_preds = []
                for pred in preds:
                    if np.isnan(pred) or np.isinf(pred) or abs(pred) > 100 * current_price:
                        cleaned_preds.append(current_price)
                    else:
                        cleaned_preds.append(pred)
                model_predictions[model_name] = cleaned_preds
        
        # 计算集成预测结果
        ensemble_preds = []
        for i in range(forecast_steps):
            weighted_sum = 0
            for model_name, preds in model_predictions.items():
                if model_name in weights and i < len(preds):
                    weighted_sum += preds[i] * weights[model_name]
            
            # 确保集成结果合理
            if np.isnan(weighted_sum) or np.isinf(weighted_sum):
                ensemble_preds.append(current_price)
            else:
                # 限制变化幅度在合理范围内（单日最大波动10%）
                max_change = current_price * 0.10
                predicted_price = current_price + weighted_sum - current_price
                if abs(predicted_price - current_price) > max_change:
                    if predicted_price > current_price:
                        predicted_price = current_price + max_change
                    else:
                        predicted_price = current_price - max_change
                
                ensemble_preds.append(predicted_price)
        
        # 保存所有预测结果
        self.predictions = model_predictions
        self.predictions['ensemble'] = ensemble_preds
        self.ensemble_predictions = ensemble_preds
        
        # 输出预测结果
        logger.info(f"未来{forecast_steps}天各模型预测结果:")
        for i in range(min(forecast_steps, 5)):  # 只显示前5天
            logger.info(f"第{i+1}天:")
            logger.info(f"  XGBoost: {model_predictions['xgboost'][i]:.6f}")
            logger.info(f"  PyTorch LSTM: {model_predictions['lstm_py'][i]:.6f}")
            logger.info(f"  PyTorch GRU: {model_predictions['gru_py'][i]:.6f}")
            logger.info(f"  PyTorch Transformer: {model_predictions['transformer_py'][i]:.6f}")
            logger.info(f"  PyTorch TFT: {model_predictions['tft_py'][i]:.6f}")
            
            if 'i_ching' in model_predictions:
                logger.info(f"  易经: {model_predictions['i_ching'][i]:.6f} ({traditional_results['i_ching']['trend']} {traditional_results['i_ching']['change_percentage']:.2f}%)")
            if 'liu_yao' in model_predictions:
                logger.info(f"  六爻: {model_predictions['liu_yao'][i]:.6f} ({traditional_results['liu_yao']['trend']} {traditional_results['liu_yao']['change_percentage']:.2f}%)")
            if 'qimen' in model_predictions:
                logger.info(f"  奇门遁甲: {model_predictions['qimen'][i]:.6f} ({traditional_results['qimen']['trend']} {traditional_results['qimen']['change_percentage']:.2f}%)")
            if 'tarot' in model_predictions:
                logger.info(f"  塔罗牌: {model_predictions['tarot'][i]:.6f} ({traditional_results['tarot']['trend']} {traditional_results['tarot']['change_percentage']:.2f}%)")
            
            logger.info(f"  集成结果: {ensemble_preds[i]:.6f}")
        
        # 计算涨跌趋势
        logger.info(f"当前基金价格: {current_price:.6f}")
        logger.info("涨跌趋势判断:")
        
        for i in range(min(forecast_steps, 5)):
            change = (ensemble_preds[i] - current_price) / current_price * 100
            trend = "上涨" if change > 0 else "下跌"
            logger.info(f"第{i+1}天: {trend} {abs(change):.2f}%")
        
        logger.info("=== 多模型集成预测完成 ===")
        
        return ensemble_preds
    
    def save_predictions(self):
        """保存当前预测记录到历史中"""
        import datetime
        
        logger.info("保存预测记录到历史...")
        
        if not self.predictions or not self.ensemble_predictions:
            logger.warning("没有可保存的预测结果")
            return None
        
        # 生成唯一预测ID
        prediction_id = f"pred_{self.prediction_id_counter}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.prediction_id_counter += 1
        
        # 创建预测记录
        prediction_record = {
            'prediction_id': prediction_id,
            'prediction_time': datetime.datetime.now(),
            'forecast_steps': len(self.ensemble_predictions),
            'current_price': self.feature_matrix['fund_close'].iloc[-1],
            'model_predictions': self.predictions,
            'ensemble_predictions': self.ensemble_predictions,
            'validation_status': 'pending'
        }
        
        # 保存到历史记录
        self.prediction_history.append(prediction_record)
        
        logger.info(f"预测记录已保存，ID: {prediction_id}")
        
        return prediction_id
    
    def run_pipeline(self, forecast_steps=5, use_real_data=True, skip_prompts=False):
        """运行完整的预测流程
        
        参数:
        forecast_steps: 预测步数
        use_real_data: 是否使用真实数据
        skip_prompts: 是否跳过用户提示
        """
        logger.info("=== 苏格拉底时间序列预测系统开始运行（进化版） ===")
        
        # 设置参数
        self.use_real_data = use_real_data
        self.skip_user_prompts = skip_prompts
        
        # 初始化验证相关属性
        logger.info("初始化验证相关属性...")
        self.validation_history = []
        self.prediction_history = []
        self.actual_results = {}
        self.model_validation_metrics = {}
        self.validation_weight_adjustments = []
        
        # 爬取多源数据
        self.crawl_data()
        
        # 进行特征工程
        success = self.feature_engineering()
        if not success:
            logger.error("特征工程失败，终止流程")
            return None
        
        # 训练预测模型
        self.train_predictors()
        
        # 评估模型性能
        self.evaluate_models()
        
        # 进行多模型集成预测
        ensemble_predictions = self.make_predictions(forecast_steps=forecast_steps)
        
        # 保存当前预测记录
        prediction_id = self.save_predictions()
        
        # 如果有可用的实际结果，进行验证（暂不实现）
        if self.actual_results:
            logger.info("跳过验证过程")
        
        # 基于验证结果更新模型权重（暂不实现）
        logger.info("跳过模型权重更新")
        
        # 保存验证历史到文件（暂不实现）
        logger.info("验证历史已保存在内存中")
        
        logger.info("=== 苏格拉底时间序列预测系统（进化版）运行完成 ===")
        
        # 如果设置了跳过提示，直接返回
        if self.skip_user_prompts:
            return {
                'ensemble_predictions': ensemble_predictions,
                'prediction_id': prediction_id,
                'current_price': self.feature_matrix['fund_close'].iloc[-1]
            }
        
        # 询问用户是否要继续（跳过实际结果输入，因为功能暂未实现）
        try:
            user_input = input("\n系统已完成预测，是否继续？(y/n): ")
        except KeyboardInterrupt:
            logger.info("用户中断，程序退出")
        except Exception as e:
            logger.warning(f"用户输入异常: {e}")
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'prediction_id': prediction_id,
            'current_price': self.feature_matrix['fund_close'].iloc[-1]
        }
    
    def run_quick(self, forecast_steps=5):
        """快速运行，跳过所有用户交互"""
        return self.run_pipeline(forecast_steps=forecast_steps, skip_prompts=True)

if __name__ == "__main__":
    # 创建系统实例
    socrates = SocratesSystem()
    
    # 检查命令行参数
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # 快速运行模式
            result = socrates.run_quick(forecast_steps=5)
            print("\n快速运行完成！")
            if result:
                print(f"当前价格: {result['current_price']:.4f}")
                print(f"预测ID: {result['prediction_id']}")
                print("未来5天预测:", result['ensemble_predictions'])
        elif sys.argv[1] == "--help":
            print("用法:")
            print("  python socrates_system.py          # 标准运行模式")
            print("  python socrates_system.py --quick  # 快速运行模式（跳过用户交互）")
            print("  python socrates_system.py --help   # 显示帮助")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        # 标准运行模式
        result = socrates.run_pipeline(forecast_steps=5)