import pandas as pd
import numpy as np
import random
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
from feature_engineering import FeatureEngineer
import logging
import time
import requests
from bs4 import BeautifulSoup
import io

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def crawl_historical_data(fund_code, max_records=1000):
    """分页爬取基金历史数据"""
    all_data = []
    page = 1
    records_per_page = 200  # 每页最大记录数
    total_records = 0
    
    logger.info(f"开始分页爬取基金{fund_code}的历史数据，目标获取{max_records}条记录")
    
    while total_records < max_records:
        # 构建分页URL
        url = f"https://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page}&per={records_per_page}&sdate=&edate="
        
        try:
            # 发送请求
            response = requests.get(url)
            response.raise_for_status()
            
            # 解析数据
            # 注意：天天基金网返回的是JavaScript变量赋值格式，需要特殊处理
            data_str = response.text
            
            # 使用正则表达式提取content中的HTML内容
            import re
            content_match = re.search(r'content:"(.*?)",records', data_str, re.DOTALL)
            if not content_match:
                logger.error("无法提取HTML内容")
                break
            
            html_content = content_match.group(1)
            
            # 替换转义字符
            html_content = html_content.replace('\\"', '"')
            html_content = html_content.replace('\\/', '/')
            html_content = html_content.replace('\\n', '\n')
            html_content = html_content.replace('\\r', '\r')
            html_content = html_content.replace('\\t', '\t')
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找表格行
            table_rows = soup.find_all('tr')
            net_value_list = []
            
            # 提取数据
            for row in table_rows[1:]:  # 跳过表头
                cells = row.find_all('td')
                if len(cells) >= 4:
                    net_value_item = {
                        '净值日期': cells[0].text.strip(),
                        '单位净值': cells[1].text.strip(),
                        '累计净值': cells[2].text.strip(),
                        '日增长率': cells[3].text.strip()
                    }
                    net_value_list.append(net_value_item)
            
            if not net_value_list:
                logger.info("没有更多数据了")
                break
            
            # 转换为DataFrame
            page_data = pd.DataFrame(net_value_list)
            
            # 数据预处理
            page_data['净值日期'] = pd.to_datetime(page_data['净值日期'])
            page_data['单位净值'] = pd.to_numeric(page_data['单位净值'], errors='coerce')
            page_data['累计净值'] = pd.to_numeric(page_data['累计净值'], errors='coerce')
            page_data['日增长率'] = pd.to_numeric(page_data['日增长率'].str.strip('%'), errors='coerce') / 100
            
            # 添加到总数据
            all_data.append(page_data)
            
            # 更新计数
            current_records = len(page_data)
            total_records += current_records
            
            logger.info(f"已爬取第{page}页，获取{current_records}条记录，累计{total_records}条记录")
            
            # 如果已达到目标记录数，停止爬取
            if total_records >= max_records:
                break
            
            # 增加页码
            page += 1
            
            # 添加延迟，避免请求过快
            time.sleep(1 + random.uniform(0, 1))
            
        except Exception as e:
            logger.error(f"爬取第{page}页数据失败：{e}")
            break
    
    if not all_data:
        raise ValueError("未能获取任何数据")
    
    # 合并所有页的数据
    full_data = pd.concat(all_data, ignore_index=True)
    
    # 去重并按时间排序
    full_data = full_data.drop_duplicates(subset=['净值日期'])
    full_data = full_data.sort_values('净值日期')
    
    logger.info(f"数据爬取完成，共获取{len(full_data)}条记录")
    logger.info(f"数据日期范围：{full_data['净值日期'].min()} 到 {full_data['净值日期'].max()}")
    
    return full_data

def calculate_confidence_interval(forecast, forecast_std, alpha=0.05):
    """计算预测的置信区间"""
    from scipy import stats
    z_score = stats.norm.ppf(1 - alpha / 2)
    lower = forecast - z_score * forecast_std
    upper = forecast + z_score * forecast_std
    return lower, upper

def predict_gold_c_enhanced_1000():
    """增强版博时黄金C预测分析（使用1000+样本）"""
    try:
        # 博时黄金C的基金代码
        fund_code = "002611"
        
        # 爬取历史数据
        full_data = crawl_historical_data(fund_code, max_records=1000)
        
        # 数据预处理
        data = full_data.set_index('净值日期')
        data = data.sort_index()
        
        # 计算历史收益率（使用对数收益率）
        data['log_return'] = np.log(data['单位净值'] / data['单位净值'].shift(1)) * 100
        data['simple_return'] = data['单位净值'].pct_change() * 100
        data['cumulative_return'] = (1 + data['simple_return'] / 100).cumprod() - 1
        
        logger.info("历史收益统计:")
        logger.info(f"平均日收益率(对数): {data['log_return'].mean():.4f}%")
        logger.info(f"日收益率标准差(对数): {data['log_return'].std():.4f}%")
        logger.info(f"最大日涨幅: {data['simple_return'].max():.4f}%")
        logger.info(f"最大日跌幅: {data['simple_return'].min():.4f}%")
        
        # 创建DataProcessor实例，爬取其他数据源
        logger.info("开始爬取其他数据源...")
        processor = DataProcessor()
        
        # 爬取黄金9999数据
        gold9999_data = processor.crawl_gold9999()
        logger.info(f"黄金9999数据获取完成，共{len(gold9999_data)}条记录")
        
        # 爬取伦敦金现数据
        gold_london_data = processor.crawl_gold_london()
        logger.info(f"伦敦金现数据获取完成，共{len(gold_london_data)}条记录")
        
        # 爬取美元兑人民币汇率数据
        usdcny_data = processor.crawl_usdcny()
        logger.info(f"美元兑人民币汇率数据获取完成，共{len(usdcny_data)}条记录")
        
        # 准备特征工程所需的数据格式
        fund_data_for_fe = data[['单位净值']].rename(columns={'单位净值': 'close'})
        
        # 检查并准备其他数据源的格式
        if '日期' in gold9999_data.columns:
            gold9999_data_for_fe = gold9999_data[['日期', '收盘价']].rename(columns={'日期': 'date', '收盘价': 'close'})
        else:
            gold9999_data_for_fe = gold9999_data[['date', 'close']]
            
        if '日期' in gold_london_data.columns:
            gold_london_data_for_fe = gold_london_data[['日期', '收盘价']].rename(columns={'日期': 'date', '收盘价': 'close'})
        else:
            gold_london_data_for_fe = gold_london_data[['date', 'close']]
            
        if '日期' in usdcny_data.columns:
            usdcny_data_for_fe = usdcny_data[['日期', '收盘价']].rename(columns={'日期': 'date', '收盘价': 'close'})
        else:
            usdcny_data_for_fe = usdcny_data[['date', 'close']]
        
        # 初始化预测器
        predictor = TimeSeriesPredictor(data, '单位净值')
        
        # 尝试多种预测模型
        logger.info("开始训练预测模型...")
        
        # 移动平均
        predictor.moving_average(window_size=10, forecast_steps=5)
        
        # 指数平滑
        predictor.exponential_smoothing(alpha=0.3, forecast_steps=5)
        
        # PMA模型（预测移动平均）
        logger.info("正在训练PMA模型...")
        predictor.predicted_moving_average(window_size=3, lookback_period=10, forecast_steps=5)
        
        # 参数调优 - 对PMA模型进行网格搜索
        logger.info("正在对PMA模型进行参数调优...")
        param_grid = {
            'window_size': [2, 3, 5, 7, 10],
            'lookback_period': [5, 10, 15, 20]
        }
        best_params, best_score = predictor.grid_search('PMA', param_grid, forecast_steps=5, cv=5)
        logger.info(f"PMA模型最优参数: {best_params}, 最小MSE: {best_score:.6f}")
        
        # 使用最优参数重新训练PMA模型
        logger.info("使用最优参数重新训练PMA模型...")
        predictor.predicted_moving_average(
            window_size=best_params['window_size'],
            lookback_period=best_params['lookback_period'],
            forecast_steps=5
        )
        
        # ARIMA
        try:
            predictor.arima(order=(2, 1, 1), forecast_steps=5)
        except Exception as e:
            logger.warning(f"ARIMA模型训练失败：{e}")
            # 尝试其他ARIMA参数
            try:
                predictor.arima(order=(1, 1, 1), forecast_steps=5)
            except Exception as e2:
                logger.warning(f"ARIMA(1,1,1)模型训练也失败：{e2}")
        
        # 尝试Holt-Winters（有足够数据时）
        if len(data) >= 60:
            try:
                predictor.holt_winters(seasonal_periods=30, forecast_steps=5)
            except Exception as e:
                logger.warning(f"Holt-Winters模型训练失败：{e}")
        
        # 评估模型
        evaluations = predictor.evaluate()
        logger.info("传统模型评估结果:")
        for model, metrics in evaluations.items():
            logger.info(f"{model}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")
        
        # 选择最优传统模型
        best_traditional_model = predictor.get_best_model()
        logger.info(f"最优传统模型: {best_traditional_model}")
        
        # 使用特征工程创建多因子特征矩阵
        logger.info("开始特征工程...")
        fe = FeatureEngineer()
        
        try:
            fe.load_data(
                fund_data=fund_data_for_fe,
                gold9999_data=gold9999_data_for_fe,
                gold_london_data=gold_london_data_for_fe,
                usdcny_data=usdcny_data_for_fe
            )
            
            # 创建特征矩阵
            feature_matrix = fe.create_feature_matrix(
                rolling_correlation_windows=[10, 20],
                momentum_windows=[5, 10, 20],
                volatility_windows=[10, 20, 30]
            )
            
            logger.info("特征矩阵创建完成，准备进行高级模型训练...")
            
            # 使用XGBoost模型
            logger.info("开始训练XGBoost模型...")
            xgb_predictions = predictor.xgboost(feature_matrix=feature_matrix, forecast_steps=5)
            
            # 使用LSTM模型
            logger.info("开始训练LSTM模型...")
            lstm_predictions = predictor.lstm(feature_matrix=feature_matrix, lookback=10, forecast_steps=5)
            
            # 更新评估结果，添加XGBoost和LSTM的评估
            if hasattr(predictor, 'evaluate_advanced_models'):
                advanced_evaluations = predictor.evaluate_advanced_models()
                evaluations.update(advanced_evaluations)
            
            # 选择最优模型（包括传统模型和高级模型）
            best_model = predictor.get_best_model()
            logger.info(f"最优模型（包括高级模型）: {best_model}")
            
        except Exception as e:
            logger.warning(f"特征工程或高级模型训练失败：{e}")
            import traceback
            traceback.print_exc()
            # 如果高级模型失败，使用传统模型的最佳模型
            best_model = best_traditional_model
            xgb_predictions = None
            lstm_predictions = None
        
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
        
        # 如果有高级模型预测结果，打印出来进行对比
        if xgb_predictions is not None:
            logger.info("\nXGBoost模型预测结果:")
            for i in range(forecast_steps):
                logger.info(f"第{i+1}天: {xgb_predictions[i]:.4f}")
        
        if lstm_predictions is not None:
            logger.info("\nLSTM模型预测结果:")
            for i in range(forecast_steps):
                logger.info(f"第{i+1}天: {lstm_predictions[i]:.4f}")
        
        # 使用ARIMA模型计算置信区间
        if best_model == 'ARIMA' and 'ARIMA' in predictor.models:
            model_fit = predictor.models['ARIMA']['model']
            arima_forecast = model_fit.forecast(steps=forecast_steps)
            
            # 计算预测标准差（基于模型残差）
            residuals = model_fit.resid
            residual_std = residuals.std()
            
            # 计算置信区间
            lower, upper = calculate_confidence_interval(arima_forecast, residual_std)
            
            logger.info("\nARIMA预测置信区间(95%):")
            for i in range(forecast_steps):
                logger.info(f"第{i+1}天: {arima_forecast[i]:.4f} [{lower[i]:.4f}, {upper[i]:.4f}]")
        
        # 风险评估
        # 使用对数收益率计算波动率
        daily_volatility = data['log_return'].std()
        # 年化波动率（使用年化因子√252）
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # 无风险利率（中国10年期国债收益率，约为2.5%）
        risk_free_rate = 2.5
        # 年化平均收益率
        annualized_return = data['log_return'].mean() * 252
        
        # 夏普比率计算（年化）
        sharpe_ratio = ((annualized_return - risk_free_rate) / annualized_volatility) if annualized_volatility != 0 else 0
        
        # 最大回撤计算
        max_drawdown = ((data['单位净值'].cummax() - data['单位净值']) / data['单位净值'].cummax()).max() * 100
        
        logger.info("\n风险评估指标:")
        logger.info(f"日波动率(对数): {daily_volatility:.4f}%")
        logger.info(f"年化波动率: {annualized_volatility:.4f}%")
        logger.info(f"年化平均收益率: {annualized_return:.4f}%")
        logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"最大回撤: {max_drawdown:.4f}%")
        
        # 模型稳定性分析
        if best_model == 'ARIMA' and 'ARIMA' in predictor.models:
            # 检查残差是否为白噪声
            residuals = predictor.models['ARIMA']['model'].resid
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
        
        # 准备返回结果
        result = {
            'fund_code': fund_code,
            'fund_name': '博时黄金C',
            'yesterday_value': yesterday_value,
            'today_prediction': today_prediction,
            'today_change': today_change,
            'multi_step_predictions': multi_step_predictions,
            'daily_changes': daily_changes,
            'cumulative_return': cumulative_return,
            'best_model': best_model,
            'best_traditional_model': best_traditional_model,
            'evaluations': evaluations,
            'data_sample_size': len(data),
            'data_date_range': {
                'start': data.index.min(),
                'end': data.index.max()
            },
            'historical_stats': {
                'avg_daily_return': data['log_return'].mean(),
                'std_daily_return': data['log_return'].std(),
                'max_daily_gain': data['simple_return'].max(),
                'max_daily_loss': data['simple_return'].min(),
                'volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'risk_assessment': {
                'volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'trading_signal': {
                'signal': signal,
                'confidence': confidence,
                'reason': f"基于{best_model}模型预测，今日预计{'上涨' if today_change > 0 else '下跌'}{abs(today_change):.2f}%"
            }
        }
        
        # 添加高级模型预测结果（如果有）
        if xgb_predictions is not None:
            result['xgb_predictions'] = xgb_predictions
        
        if lstm_predictions is not None:
            result['lstm_predictions'] = lstm_predictions
            
        # 添加特征工程信息
        if 'fe' in locals() and hasattr(fe, 'feature_matrix'):
            result['feature_engineering'] = {
                'feature_count': len(fe.feature_matrix.columns) if fe.feature_matrix is not None else 0,
                'sample_count_after_fe': len(fe.feature_matrix) if fe.feature_matrix is not None else 0
            }
        
        return result
        
    except Exception as e:
        logger.error(f"预测分析失败：{e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    result = predict_gold_c_enhanced_1000()
    
    print("\n" + "="*70)
    print(f"博时黄金C({result['fund_code']})增强版预测分析")
    print(f"样本量: {result['data_sample_size']} 条记录")
    print(f"数据日期范围: {result['data_date_range']['start']} 到 {result['data_date_range']['end']}")
    print("="*70)
    
    print("\n【基本预测信息】")
    print(f"昨日净值：{result['yesterday_value']:.4f}")
    print(f"今日预测净值：{result['today_prediction']:.4f}")
    print(f"今日预测涨跌幅：{result['today_change']:.2f}%")
    
    print("\n【未来5天预测】")
    for i in range(len(result['multi_step_predictions'])):
        print(f"第{i+1}天：{result['multi_step_predictions'][i]:.4f} (涨跌幅：{result['daily_changes'][i]:.2f}%)")
    
    print(f"\n5天累计收益率：{result['cumulative_return']:.2f}%")
    
    # 打印高级模型预测结果（如果有）
    if 'xgb_predictions' in result:
        print("\n【XGBoost模型预测】")
        for i in range(len(result['xgb_predictions'])):
            print(f"第{i+1}天：{result['xgb_predictions'][i]:.4f}")
    
    if 'lstm_predictions' in result:
        print("\n【LSTM模型预测】")
        for i in range(len(result['lstm_predictions'])):
            print(f"第{i+1}天：{result['lstm_predictions'][i]:.4f}")
    
    print("\n【模型性能】")
    print(f"最优传统模型：{result['best_traditional_model']}")
    print(f"最优模型：{result['best_model']}")
    print("\n各模型评估结果：")
    for model, metrics in result['evaluations'].items():
        print(f"{model}：RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")
    
    # 打印特征工程信息（如果有）
    if 'feature_engineering' in result:
        print("\n【特征工程】")
        print(f"生成特征数量：{result['feature_engineering']['feature_count']}")
        print(f"特征工程后样本量：{result['feature_engineering']['sample_count_after_fe']}")
    
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
