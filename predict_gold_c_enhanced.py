import pandas as pd
import numpy as np
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
import logging
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_confidence_interval(forecast, forecast_std, alpha=0.05):
    """计算预测的置信区间"""
    from scipy import stats
    z_score = stats.norm.ppf(1 - alpha / 2)
    lower = forecast - z_score * forecast_std
    upper = forecast + z_score * forecast_std
    return lower, upper

def predict_gold_c_enhanced():
    """增强版博时黄金C预测分析"""
    try:
        # 博时黄金C的基金代码
        fund_code = "002611"
        
        # 构建天天基金网的历史净值数据URL
        url = f"https://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=365&sdate=&edate="
        
        logger.info(f"开始爬取博时黄金C({fund_code})的历史净值数据")
        
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 爬取数据
        try:
            data = processor.load_data(
                source=url,
                time_column='净值日期',
                target_column='单位净值',
                source_type='web',
                crawl_mode='basic'
            )
        except Exception as e:
            logger.warning(f"基本爬取失败：{e}，尝试使用浏览器驱动爬取")
            data = processor.load_data(
                source=f"https://fund.eastmoney.com/{fund_code}.html",
                time_column='净值日期',
                target_column='单位净值',
                source_type='web',
                crawl_mode='web_driver',
                browser_type='firefox',
                headless=True,
                table_selector='#bodydiv > div > div > div.basic-new > div.bs_jz > div.col-left > div > div.dataList > div > table'
            )
        
        logger.info(f"成功获取数据，数据形状: {data.shape}")
        logger.info(f"数据日期范围: {data.index.min()} 到 {data.index.max()}")
        
        # 数据预处理
        data = data.sort_index()
        
        # 计算历史收益率
        data['daily_return'] = data['单位净值'].pct_change() * 100
        data['cumulative_return'] = (1 + data['daily_return'] / 100).cumprod() - 1
        
        logger.info("历史收益统计:")
        logger.info(f"平均日收益率: {data['daily_return'].mean():.4f}%")
        logger.info(f"日收益率标准差: {data['daily_return'].std():.4f}%")
        logger.info(f"最大日涨幅: {data['daily_return'].max():.4f}%")
        logger.info(f"最大日跌幅: {data['daily_return'].min():.4f}%")
        
        # 初始化预测器
        predictor = TimeSeriesPredictor(data, '单位净值')
        
        # 尝试多种预测模型
        logger.info("开始训练预测模型...")
        
        # 移动平均
        predictor.moving_average(window_size=5, forecast_steps=5)
        
        # 指数平滑
        predictor.exponential_smoothing(alpha=0.3, forecast_steps=5)
        
        # ARIMA
        try:
            predictor.arima(order=(1, 1, 1), forecast_steps=5)
        except Exception as e:
            logger.warning(f"ARIMA模型训练失败：{e}")
        
        # 评估模型
        evaluations = predictor.evaluate()
        logger.info("模型评估结果:")
        for model, metrics in evaluations.items():
            logger.info(f"{model}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")
        
        # 选择最优模型
        best_model = predictor.get_best_model()
        logger.info(f"最优模型: {best_model}")
        
        # 使用最优模型进行多步预测（未来5天）
        forecast_steps = 5
        multi_step_predictions = predictor.predict_with_best_model(forecast_steps=forecast_steps)
        
        logger.info(f"未来{forecast_steps}天预测结果:")
        for i in range(forecast_steps):
            logger.info(f"第{i+1}天: {multi_step_predictions[i]:.4f}")
        
        # 计算预测的涨跌幅
        yesterday_value = data['单位净值'].iloc[-1]
        daily_changes = []
        cumulative_change = 1.0
        
        for i in range(forecast_steps):
            if i == 0:
                change = (multi_step_predictions[i] - yesterday_value) / yesterday_value * 100
            else:
                change = (multi_step_predictions[i] - multi_step_predictions[i-1]) / multi_step_predictions[i-1] * 100
            daily_changes.append(change)
            cumulative_change *= (1 + change / 100)
        
        cumulative_return = (cumulative_change - 1) * 100
        
        # 使用ARIMA模型计算置信区间
        if best_model == 'ARIMA':
            model_fit = predictor.models['ARIMA']['model']
            arima_forecast = model_fit.forecast(steps=forecast_steps)
            
            # 计算预测标准差（基于模型残差）
            residuals = model_fit.resid
            residual_std = residuals.std()
            
            # 计算置信区间
            lower, upper = calculate_confidence_interval(arima_forecast, residual_std)
            
            logger.info("\nARIMA预测置信区间(95%):")
            for i in range(forecast_steps):
                logger.info(f"第{i+1}天: {arima_forecast.iloc[i]:.4f} [{lower.iloc[i]:.4f}, {upper.iloc[i]:.4f}]")
        
        # 风险评估
        historical_volatility = data['daily_return'].std()
        sharpe_ratio = (data['daily_return'].mean() / historical_volatility) if historical_volatility != 0 else 0
        max_drawdown = ((data['单位净值'].cummax() - data['单位净值']) / data['单位净值'].cummax()).max() * 100
        
        logger.info("\n风险评估指标:")
        logger.info(f"历史波动率: {historical_volatility:.4f}%")
        logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"最大回撤: {max_drawdown:.4f}%")
        
        # 模型稳定性分析
        if best_model == 'ARIMA':
            # 检查残差是否为白噪声
            residuals = model_fit.resid
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            logger.info(f"\n模型稳定性测试:")
            logger.info(f"Ljung-Box检验p值: {lb_test['lb_pvalue'].iloc[0]:.6f}")
            if lb_test['lb_pvalue'].iloc[0] > 0.05:
                logger.info("残差序列为白噪声，模型拟合良好")
            else:
                logger.info("残差序列存在自相关性，模型可能需要改进")
        
        # 交易信号生成
        today_prediction = multi_step_predictions[0]
        today_change = daily_changes[0]
        
        # 基于预测和风险指标生成交易建议
        if today_change > 0.5:
            signal = "强烈买入"
            confidence = "高"
        elif today_change > 0.1:
            signal = "买入"
            confidence = "中"
        elif today_change < -0.5:
            signal = "强烈卖出"
            confidence = "高"
        elif today_change < -0.1:
            signal = "卖出"
            confidence = "中"
        else:
            signal = "持有"
            confidence = "低"
        
        logger.info(f"\n交易信号:")
        logger.info(f"今日预测涨跌幅: {today_change:.2f}%")
        logger.info(f"建议操作: {signal}")
        logger.info(f"信号置信度: {confidence}")
        
        return {
            'fund_code': fund_code,
            'fund_name': '博时黄金C',
            'yesterday_value': yesterday_value,
            'today_prediction': today_prediction,
            'today_change': today_change,
            'multi_step_predictions': multi_step_predictions,
            'daily_changes': daily_changes,
            'cumulative_return': cumulative_return,
            'best_model': best_model,
            'evaluations': evaluations,
            'historical_stats': {
                'avg_daily_return': data['daily_return'].mean(),
                'std_daily_return': data['daily_return'].std(),
                'max_daily_gain': data['daily_return'].max(),
                'max_daily_loss': data['daily_return'].min(),
                'volatility': historical_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'risk_assessment': {
                'volatility': historical_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'trading_signal': {
                'signal': signal,
                'confidence': confidence,
                'reason': f"基于{best_model}模型预测，今日预计{'上涨' if today_change > 0 else '下跌'}{abs(today_change):.2f}%"
            }
        }
        
    except Exception as e:
        logger.error(f"预测分析失败：{e}")
        raise

if __name__ == "__main__":
    result = predict_gold_c_enhanced()
    
    print("\n" + "="*70)
    print(f"博时黄金C({result['fund_code']})增强版预测分析")
    print("="*70)
    
    print("\n【基本预测信息】")
    print(f"昨日净值：{result['yesterday_value']:.4f}")
    print(f"今日预测净值：{result['today_prediction']:.4f}")
    print(f"今日预测涨跌幅：{result['today_change']:.2f}%")
    
    print("\n【未来5天预测】")
    for i in range(len(result['multi_step_predictions'])):
        print(f"第{i+1}天：{result['multi_step_predictions'][i]:.4f} (涨跌幅：{result['daily_changes'][i]:.2f}%)")
    
    print(f"\n5天累计收益率：{result['cumulative_return']:.2f}%")
    
    print("\n【模型性能】")
    print(f"最优模型：{result['best_model']}")
    for model, metrics in result['evaluations'].items():
        print(f"{model}：RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")
    
    print("\n【风险评估】")
    print(f"历史波动率：{result['risk_assessment']['volatility']:.4f}%")
    print(f"夏普比率：{result['risk_assessment']['sharpe_ratio']:.4f}")
    print(f"最大回撤：{result['risk_assessment']['max_drawdown']:.4f}%")
    
    print("\n【交易建议】")
    print(f"建议操作：{result['trading_signal']['signal']}")
    print(f"信号置信度：{result['trading_signal']['confidence']}")
    print(f"建议理由：{result['trading_signal']['reason']}")
    
    print("\n" + "="*70)
    print("⚠️  风险提示：本预测基于历史数据的统计模型，不构成投资建议")
    print("⚠️  投资有风险，入市需谨慎")
    print("="*70)
