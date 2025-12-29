import pandas as pd
import numpy as np
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_gold_c():
    """预测博时黄金C今日是否大涨"""
    try:
        # 博时黄金C的基金代码
        fund_code = "002611"
        
        # 构建天天基金网的历史净值数据URL
        # 天天基金网：https://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=002611&page=1&per=60&sdate=&edate=
        url = f"https://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=365&sdate=&edate="
        
        logger.info(f"开始爬取博时黄金C({fund_code})的历史净值数据")
        
        # 初始化数据处理器
        processor = DataProcessor()
        
        # 爬取数据
        # 注意：天天基金网的数据是JSON格式，需要特殊处理
        try:
            # 首先尝试基本爬取
            data = processor.load_data(
                source=url,
                time_column='净值日期',
                target_column='单位净值',
                source_type='web',
                crawl_mode='basic'
            )
        except Exception as e:
            logger.warning(f"基本爬取失败：{e}，尝试使用浏览器驱动爬取")
            # 如果基本爬取失败，尝试使用浏览器驱动爬取
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
        logger.info(f"数据示例:\n{data.head()}")
        
        # 数据预处理
        # 确保数据按时间排序
        data = data.sort_index()
        
        # 初始化预测器
        predictor = TimeSeriesPredictor(data, '单位净值')
        
        # 尝试多种预测模型
        logger.info("开始训练预测模型...")
        
        # 移动平均
        predictor.moving_average(window_size=5, forecast_steps=1)
        
        # 指数平滑
        predictor.exponential_smoothing(alpha=0.3, forecast_steps=1)
        
        # ARIMA
        try:
            predictor.arima(order=(1, 1, 1), forecast_steps=1)
        except Exception as e:
            logger.warning(f"ARIMA模型训练失败：{e}")
        
        # Holt-Winters
        try:
            predictor.holt_winters(seasonal_periods=30, forecast_steps=1)
        except Exception as e:
            logger.warning(f"Holt-Winters模型训练失败：{e}")
        
        # 评估模型
        evaluations = predictor.evaluate()
        logger.info("模型评估结果:")
        for model, metrics in evaluations.items():
            logger.info(f"{model}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")
        
        # 选择最优模型
        best_model = predictor.get_best_model()
        logger.info(f"最优模型: {best_model}")
        
        # 使用最优模型预测今日净值
        today_prediction = predictor.predict_with_best_model(forecast_steps=1)[0]
        logger.info(f"今日预测净值: {today_prediction:.4f}")
        
        # 获取昨日净值
        yesterday_value = data['单位净值'].iloc[-1]
        logger.info(f"昨日净值: {yesterday_value:.4f}")
        
        # 计算涨跌幅
        change = (today_prediction - yesterday_value) / yesterday_value * 100
        logger.info(f"预测涨跌幅: {change:.2f}%")
        
        # 判断是否大涨（这里定义涨跌幅超过0.5%为大涨）
        if change > 0.5:
            result = "大涨"
        elif change > 0:
            result = "上涨"
        elif change < -0.5:
            result = "大跌"
        elif change < 0:
            result = "下跌"
        else:
            result = "持平"
        
        logger.info(f"博时黄金C今日预测：{result}")
        
        return {
            'fund_code': fund_code,
            'fund_name': '博时黄金C',
            'yesterday_value': yesterday_value,
            'today_prediction': today_prediction,
            'change_percentage': change,
            'prediction_result': result,
            'best_model': best_model,
            'evaluations': evaluations
        }
        
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise

if __name__ == "__main__":
    result = predict_gold_c()
    print("\n" + "="*50)
    print(f"博时黄金C({result['fund_code']})今日预测结果：")
    print(f"昨日净值：{result['yesterday_value']:.4f}")
    print(f"今日预测净值：{result['today_prediction']:.4f}")
    print(f"预测涨跌幅：{result['change_percentage']:.2f}%")
    print(f"预测结果：{result['prediction_result']}")
    print(f"使用模型：{result['best_model']}")
    print("="*50)
