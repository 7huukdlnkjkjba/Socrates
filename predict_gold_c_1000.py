import pandas as pd
import numpy as np
import random
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
from feature_engineering import FeatureEngineer
from risk_management import RiskManagement
from backtesting import BacktestingSystem
from stress_testing import StressTesting
import logging
import time
import requests
from bs4 import BeautifulSoup
import io

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pickle
import os
import hashlib
from datetime import datetime, timedelta

def get_cache_file_path(fund_code):
    """生成缓存文件路径"""
    cache_dir = 'data_cache'
    # 使用基金代码生成唯一的缓存文件名
    cache_filename = f"{fund_code}_historical_data.pkl"
    return os.path.join(cache_dir, cache_filename)

def is_cache_valid(cache_file, max_age_days=7):
    """检查缓存是否有效
    
    参数:
    cache_file: 缓存文件路径
    max_age_days: 缓存最大有效天数
    
    返回:
    bool: 缓存是否有效
    """
    if not os.path.exists(cache_file):
        return False
    
    # 检查文件修改时间
    file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
    now = datetime.now()
    age = now - file_mtime
    
    return age <= timedelta(days=max_age_days)

def save_to_cache(data, cache_file):
    """保存数据到缓存"""
    # 确保缓存目录存在
    cache_dir = os.path.dirname(cache_file)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 保存数据到文件
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"数据已保存到缓存：{cache_file}")

def load_from_cache(cache_file):
    """从缓存加载数据"""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"从缓存加载数据：{cache_file}")
        return data
    except Exception as e:
        logger.error(f"加载缓存失败：{e}")
        return None

def crawl_historical_data(fund_code, max_records=1000, use_cache=True, cache_max_age_days=7):
    """分页爬取基金历史数据，优先使用本地缓存
    
    参数:
    fund_code: 基金代码
    max_records: 最大记录数
    use_cache: 是否使用缓存
    cache_max_age_days: 缓存最大有效天数
    
    返回:
    pd.DataFrame: 基金历史数据
    """
    # 生成缓存文件路径
    cache_file = get_cache_file_path(fund_code)
    
    # 检查是否使用缓存且缓存有效
    if use_cache and is_cache_valid(cache_file, cache_max_age_days):
        # 从缓存加载数据
        cached_data = load_from_cache(cache_file)
        if cached_data is not None:
            # 检查缓存数据是否满足记录数要求
            if len(cached_data) >= max_records:
                logger.info(f"使用缓存数据，共{len(cached_data)}条记录，满足{max_records}条记录要求")
                return cached_data
            else:
                logger.info(f"缓存数据记录数不足，需要重新爬取")
        else:
            logger.info("缓存数据无效，需要重新爬取")
    
    # 缓存无效或不使用缓存，执行爬取
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

def check_gold_etf_characteristics(predictions, last_price, hist_data):
    """
    检查预测结果是否符合黄金ETF的实际波动特性
    
    参数:
    predictions: 预测值列表
    last_price: 昨日实际价格
    hist_data: 历史数据
    
    返回:
    dict: 检查结果
    """
    results = {
        "has_issues": False,
        "issues": [],
        "recommendations": []
    }
    
    # 计算涨跌幅
    returns = []
    for i, pred in enumerate(predictions):
        if i == 0:
            prev_price = last_price
        else:
            prev_price = predictions[i-1]
        return_val = (pred - prev_price) / prev_price
        returns.append(return_val)
    
    # 黄金ETF特性检查
    # 1. 单日涨跌幅检查（黄金极少超过±3%）
    max_abs_return = max(abs(r) for r in returns)
    if max_abs_return > 0.03:
        results["has_issues"] = True
        results["issues"].append(f"黄金单日涨跌幅超过3%，现实中极为罕见: {max_abs_return*100:.2f}%")
    
    # 2. 连续大幅波动检查（黄金极少连续两天大幅波动）
    consecutive_large_moves = 0
    for i in range(1, len(returns)):
        if abs(returns[i]) > 0.02 and abs(returns[i-1]) > 0.02:
            consecutive_large_moves += 1
    if consecutive_large_moves > 0:
        results["has_issues"] = True
        results["issues"].append(f"黄金连续大幅波动，现实中极为罕见: {consecutive_large_moves}次连续大幅波动")
    
    # 3. 波动率检查（黄金日波动率通常在0.5%-1.5%之间）
    pred_vol = np.std(returns) * 100  # 转换为百分比
    if pred_vol > 2.0:
        results["has_issues"] = True
        results["issues"].append(f"黄金预测波动率过高: {pred_vol:.2f}%（正常范围：0.5%-1.5%）")
    
    # 4. 累计涨跌幅检查（黄金5天累计涨跌幅极少超过±5%）
    cumulative_return = (predictions[-1] - last_price) / last_price
    if abs(cumulative_return) > 0.05:
        results["has_issues"] = True
        results["issues"].append(f"黄金5天累计涨跌幅超过5%，现实中较为罕见: {cumulative_return*100:.2f}%")
    
    # 5. 趋势合理性检查（黄金作为避险资产，通常不会持续单边大幅下跌）
    all_neg = all(r < -0.01 for r in returns)
    all_pos = all(r > 0.01 for r in returns)
    if all_neg or all_pos:
        results["has_issues"] = True
        results["issues"].append(f"黄金连续5天单边大幅{'下跌' if all_neg else '上涨'}，现实中极为罕见")
    
    # 生成建议
    if results["has_issues"]:
        results["recommendations"].append("建议进一步调整预测模型，考虑黄金ETF的实际波动特性")
        results["recommendations"].append("可以尝试降低模型复杂度，减少过度拟合")
        results["recommendations"].append("考虑加入外部特征（美元指数、美债收益率等）")
    
    return results

def sanity_check(predictions, last_price, hist_data, model_name="未知模型"):
    """
    预测结果合理性检查
    
    参数:
    predictions: 预测值列表
    last_price: 昨日实际价格
    hist_data: 历史数据
    model_name: 模型名称
    
    返回:
    dict: 检查结果
    """
    results = {
        "model": model_name,
        "has_errors": False,
        "errors": [],
        "warnings": [],
        "checks": []
    }
    
    # 1. 检查预测值是否非负
    if any(p < 0 for p in predictions):
        results["has_errors"] = True
        results["errors"].append(f"预测值出现负值: {min(predictions):.6f}")
    else:
        results["checks"].append("✓ 所有预测值均为非负")
    
    # 2. 计算涨跌幅
    returns = []
    for i, pred in enumerate(predictions):
        if i == 0:
            prev_price = last_price
        else:
            prev_price = predictions[i-1]
        return_val = (pred - prev_price) / prev_price
        returns.append(return_val)
    
    # 3. 检查涨跌幅是否在合理范围内（黄金ETF特化，使用3倍标准差原则）
    hist_returns = hist_data['simple_return'].dropna() / 100  # 转换为小数
    mean_return = hist_returns.mean()
    std_return = hist_returns.std()
    
    max_allowed = mean_return + 3 * std_return
    min_allowed = mean_return - 3 * std_return
    
    max_return = max(returns)
    min_return = min(returns)
    
    if max_return > max_allowed:
        results["warnings"].append(f"预测单日最大涨跌幅超过3倍标准差: {max_return*100:.2f}% (允许: {max_allowed*100:.2f}%)")
    if min_return < min_allowed:
        results["warnings"].append(f"预测单日最小涨跌幅超过3倍标准差: {min_return*100:.2f}% (允许: {min_allowed*100:.2f}%)")
    else:
        results["checks"].append("✓ 涨跌幅在3倍标准差范围内")
    
    # 4. 检查波动率是否合理（黄金ETF特化，波动率不超过历史3倍）
    pred_vol = np.std(returns) * 100  # 转换为百分比
    hist_vol = hist_data['simple_return'].std()
    
    if pred_vol > hist_vol * 3:
        results["warnings"].append(f"预测波动率超过历史波动率3倍: {pred_vol:.2f}% (历史: {hist_vol:.2f}%)")
    else:
        results["checks"].append("✓ 波动率在合理范围内")
    
    # 5. 检查预测值是否与历史价格偏离过大
    avg_hist_price = hist_data['单位净值'].mean()
    avg_pred_price = np.mean(predictions)
    price_diff_pct = abs(avg_pred_price - avg_hist_price) / avg_hist_price * 100
    
    if price_diff_pct > 20:  # 放宽到20%，因为黄金价格可能长期趋势变化
        results["warnings"].append(f"预测平均价格与历史平均价格偏离超过20%: 历史平均{avg_hist_price:.4f}, 预测平均{avg_pred_price:.4f}")
    else:
        results["checks"].append("✓ 预测价格与历史价格偏离在合理范围内")
    
    # 6. 黄金ETF特性检查
    gold_characteristics = check_gold_etf_characteristics(predictions, last_price, hist_data)
    if gold_characteristics["has_issues"]:
        for issue in gold_characteristics["issues"]:
            results["warnings"].append(f"黄金ETF特性检查: {issue}")
    else:
        results["checks"].append("✓ 符合黄金ETF实际波动特性")
    
    return results

def multi_scale_gold_forecast(predictor, data, forecast_steps=5, feature_matrix=None):
    """多时间尺度黄金预测融合
    
    参数:
    predictor: TimeSeriesPredictor实例
    data: 历史数据
    forecast_steps: 预测步数
    feature_matrix: 特征矩阵
    
    返回:
    list: 融合后的预测值列表
    """
    logger.info("\n=== 开始多时间尺度融合预测 ===")
    
    # 1. 日度预测（中频，主预测）
    logger.info("  生成日度预测...")
    daily_predictions = predictor.lstm(lookback=10, forecast_steps=forecast_steps, feature_matrix=feature_matrix, use_custom_loss=True)
    daily_weight = 0.5
    
    # 2. 周度预测（低频，趋势性强）
    logger.info("  生成周度预测...")
    # 使用Transformer模型进行周度预测，使用更长的lookback周期
    weekly_predictions = predictor.transformer(lookback=30, forecast_steps=forecast_steps, feature_matrix=feature_matrix)
    weekly_weight = 0.3
    
    # 3. 移动平均预测（高频，作为补充）
    logger.info("  生成移动平均预测...")
    ma_predictions = predictor.moving_average(window_size=5, forecast_steps=forecast_steps)
    ma_weight = 0.2
    
    # 4. 融合预测结果
    logger.info("  融合多时间尺度预测结果...")
    fused_predictions = []
    for i in range(forecast_steps):
        # 对每个时间步进行融合
        fused = (
            daily_predictions[i] * daily_weight +
            weekly_predictions[i] * weekly_weight * 1.5 +  # 低频趋势加强
            ma_predictions[i] * ma_weight * 0.5  # 高频预测打5折
        )
        fused_predictions.append(fused)
    
    # 5. 应用安全约束
    last_price = data['单位净值'].iloc[-1]
    constrained_fused = apply_safety_constraints(fused_predictions, last_price, data)
    
    logger.info("=== 多时间尺度融合预测完成 ===")
    return constrained_fused

def apply_safety_constraints(predictions, last_price, hist_data):
    """
    应用安全约束到预测结果，针对黄金ETF的特性进行优化
    
    参数:
    predictions: 原始预测值列表
    last_price: 昨日实际价格
    hist_data: 历史数据
    
    返回:
    list: 应用约束后的预测值列表
    """
    # 计算历史涨跌幅统计
    hist_returns = hist_data['simple_return'].dropna() / 100  # 转换为小数
    mean_return = hist_returns.mean()
    std_return = hist_returns.std()
    
    # 黄金ETF特定约束：使用3倍标准差原则
    # 严格遵循用户建议：黄金日波动率通常在0.5%-2%之间
    three_sigma_upper = mean_return + 3 * std_return
    three_sigma_lower = mean_return - 3 * std_return
    
    # 更严格的资产特性约束：黄金ETF正常日波动范围±2%
    asset_specific_upper = 0.02
    asset_specific_lower = -0.02
    
    # 最终约束：取更严格的边界
    max_allowed_return = min(three_sigma_upper, asset_specific_upper)
    min_allowed_return = max(three_sigma_lower, asset_specific_lower)
    
    # 打印约束信息
    logger.info(f"黄金ETF安全约束参数：")
    logger.info(f"  平均日收益率: {mean_return*100:.4f}%")
    logger.info(f"  日收益率标准差: {std_return*100:.4f}%")
    logger.info(f"  3倍标准差范围: [{three_sigma_lower*100:.2f}%, {three_sigma_upper*100:.2f}%]")
    logger.info(f"  资产特性范围: [{asset_specific_lower*100:.2f}%, {asset_specific_upper*100:.2f}%]")
    logger.info(f"  最终日约束: [{min_allowed_return*100:.2f}%, {max_allowed_return*100:.2f}%]")
    logger.info(f"  5天累计约束: ±3%")
    
    # 获取原始预测的第一天涨跌幅
    first_pred = predictions[0]
    first_return = (first_pred - last_price) / last_price
    
    # 应用安全约束到第一天涨跌幅
    if abs(first_return) > max(abs(max_allowed_return), abs(min_allowed_return)):
        logger.warning(f"原始第一天预测涨跌幅{first_return*100:.2f}%超过安全边界，进行修正")
        # 按比例缩放
        safe_first_return = np.clip(first_return, min_allowed_return, max_allowed_return)
    else:
        safe_first_return = first_return
    
    # 生成5天预测（考虑严格的波动衰减）
    constrained_preds = []
    current_price = last_price
    
    for i in range(len(predictions)):
        # 第i天的波动：使用更严格的衰减系数（每天衰减30%）
        # 严格遵循用户建议：黄金连续大幅波动极为罕见，波动应快速衰减
        decay_factor = 0.7 ** i  # 衰减系数，每天衰减30%
        
        # 计算当天的涨跌幅：第一天是基础，后续天逐渐衰减
        if i == 0:
            day_return = safe_first_return
        else:
            # 仅衰减波动，移除反向修正（不符合黄金实际波动特性）
            day_return = safe_first_return * decay_factor
        
        # 再次确保涨跌幅在安全范围内
        day_return = np.clip(day_return, min_allowed_return, max_allowed_return)
        
        # 计算预测价格
        current_price = current_price * (1 + day_return)
        
        # 确保价格非负
        current_price = max(current_price, 0.0001)
        
        constrained_preds.append(current_price)
    
    # 检查5天累计涨跌幅是否合理（严格控制在±3%以内）
    five_day_return = (constrained_preds[-1] - last_price) / last_price
    if abs(five_day_return) > 0.03:
        logger.warning(f"5天累计预测涨跌幅{five_day_return*100:.2f}%超过安全边界±3%，进行二次修正")
        
        # 计算需要的调整比例
        target_return = np.clip(five_day_return, -0.03, 0.03)
        scale_factor = target_return / five_day_return if five_day_return != 0 else 1.0
        
        # 对每一天的涨跌幅进行调整，保持趋势但控制累计幅度
        adjusted_preds = []
        adjusted_price = last_price
        for i in range(len(constrained_preds)):
            if i == 0:
                prev_price = last_price
            else:
                prev_price = constrained_preds[i-1]
            
            # 计算原始约束后的当天涨跌幅
            orig_day_return = (constrained_preds[i] - prev_price) / prev_price
            
            # 应用调整比例
            adjusted_return = orig_day_return * scale_factor
            
            # 再次确保涨跌幅在安全范围内
            adjusted_return = np.clip(adjusted_return, min_allowed_return, max_allowed_return)
            
            # 计算调整后的价格
            adjusted_price = adjusted_price * (1 + adjusted_return)
            adjusted_preds.append(adjusted_price)
        
        constrained_preds = adjusted_preds
    
    return constrained_preds

class GoldTradingAdvisor:
    """黄金交易建议生成器"""
    
    def __init__(self, risk_free_rate=0.025):
        """
        初始化交易建议生成器
        
        参数:
        risk_free_rate: 无风险收益率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def generate_signal(self, prediction, uncertainty_metrics, current_price, hist_data):
        """
        生成交易信号和止损止盈建议
        
        参数:
        prediction: 预测结果字典，包含mean、median等
        uncertainty_metrics: 不确定性指标字典，包含std、probability_up等
        current_price: 当前价格
        hist_data: 历史数据
        
        返回:
        dict: 包含交易信号、置信度、仓位大小、止损止盈等信息
        """
        # 1. 基础信号计算
        expected_return = prediction['mean'][0] - current_price
        expected_pct = expected_return / current_price
        
        # 2. 计算历史波动率（用于动态设置止损止盈）
        hist_returns = hist_data['simple_return'].dropna() / 100  # 转换为小数
        hist_volatility = hist_returns.std() * np.sqrt(252)  # 年化波动率
        daily_volatility = hist_returns.std()  # 日波动率
        
        # 3. 考虑不确定性后的调整
        # 使用标准差和预期最大亏损来调整风险
        risk_adjusted_return = expected_pct * (1 - abs(uncertainty_metrics.get('expected_max_loss', 0)))
        
        # 4. 生成具体建议
        signal = "持有"
        confidence = "低"
        position_size = 0.0
        
        # 计算上涨概率
        probability_up = uncertainty_metrics.get('probability_up', [0.5])[0]
        
        # 计算风险收益比
        stop_loss_pct = max(0.02, daily_volatility * 2)  # 至少2%止损，或2倍日波动率
        take_profit_pct = abs(expected_pct) * 1.5  # 止盈是预期收益的1.5倍，确保风险收益比至少1:1.5
        
        # 确保风险收益比至少为1:1.5
        if take_profit_pct / stop_loss_pct < 1.5:
            take_profit_pct = stop_loss_pct * 1.5
        
        # 生成交易信号
        if risk_adjusted_return > 0.005:  # 预期收益 > 0.5%
            if probability_up > 0.65:  # 上涨概率 > 65%
                signal = "买入"
                confidence = "高"
                position_size = min(0.3, risk_adjusted_return * 10)  # 仓位控制在30%以内
            else:
                signal = "谨慎买入"
                confidence = "中"
                position_size = risk_adjusted_return * 5
        elif risk_adjusted_return < -0.005:  # 预期亏损 > 0.5%
            signal = "卖出"
            confidence = "高"
            position_size = 0
        else:
            signal = "持有"
            confidence = "低"
            position_size = 0
        
        # 计算止损止盈价格
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        
        # 确保止损价格非负
        stop_loss = max(stop_loss, 0.0001)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'position_size': f"{position_size*100:.1f}%",
            'expected_return': f"{expected_pct*100:.2f}%",
            'probability_up': f"{probability_up*100:.1f}%",
            'stop_loss': f"{stop_loss:.4f}",
            'take_profit': f"{take_profit:.4f}",
            'risk_reward_ratio': f"{take_profit_pct/stop_loss_pct:.2f}:1",
            'daily_volatility': f"{daily_volatility*100:.4f}%",
            'annual_volatility': f"{hist_volatility*100:.4f}%"
        }

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
        
        # 检查数据
        logger.info("\n=== 数据检查 ===")
        logger.info(f"基金数据列：{data.columns}")
        logger.info(f"基金数据形状：{data.shape}")
        logger.info(f"黄金9999数据列：{gold9999_data.columns}")
        logger.info(f"黄金9999数据形状：{gold9999_data.shape}")
        logger.info(f"伦敦金数据列：{gold_london_data.columns}")
        logger.info(f"伦敦金数据形状：{gold_london_data.shape}")
        logger.info(f"美元兑人民币数据列：{usdcny_data.columns}")
        logger.info(f"美元兑人民币数据形状：{usdcny_data.shape}")
        
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
        
        # 训练LSTM模型，使用带有物理约束的自定义损失函数
        try:
            logger.info("正在训练LSTM模型...")
            predictor.lstm(lookback=10, forecast_steps=5, feature_matrix=fund_data_for_fe, use_custom_loss=True)
        except Exception as e:
            logger.warning(f"LSTM模型训练失败：{e}")
            import traceback
            traceback.print_exc()
        
        # 训练Transformer模型
        try:
            logger.info("正在训练Transformer模型...")
            predictor.transformer(lookback=10, forecast_steps=5, feature_matrix=fund_data_for_fe)
        except Exception as e:
            logger.warning(f"Transformer模型训练失败：{e}")
            import traceback
            traceback.print_exc()
        
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
            
            # 创建精简的核心特征矩阵（8个核心特征）
            feature_matrix = fe.create_core_feature_matrix()
            
            logger.info("特征矩阵创建完成，准备进行高级模型训练...")
            logger.info(f"特征矩阵列：{feature_matrix.columns}")
            
            # 修复特征矩阵：添加原始目标列
            feature_matrix['单位净值'] = data.loc[feature_matrix.index, '单位净值']
            
            # 使用XGBoost模型
            logger.info("开始训练XGBoost模型...")
            xgb_predictions = predictor.xgboost(feature_matrix=feature_matrix, forecast_steps=5)
            
            # 获取LSTM和Transformer的预测结果
            lstm_predictions = predictor.predictions.get('LSTM', None)
            transformer_predictions = predictor.predictions.get('Transformer', None)
            
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
            lstm_predictions = predictor.predictions.get('LSTM', None)
            transformer_predictions = predictor.predictions.get('Transformer', None)
        
        # 生成多步预测（未来5天）
        forecast_steps = 5
        
        # 首先获取最优模型的预测
        multi_step_predictions = predictor.predict_with_best_model(forecast_steps=forecast_steps)
        
        # 然后尝试多时间尺度融合预测
        try:
            fused_predictions = multi_scale_gold_forecast(
                predictor=predictor,
                data=data,
                forecast_steps=forecast_steps,
                feature_matrix=feature_matrix if 'feature_matrix' in locals() else None
            )
            # 使用融合后的预测结果作为最终预测
            multi_step_predictions = fused_predictions
        except Exception as e:
            logger.warning(f"多时间尺度融合预测失败，使用最优模型的预测结果: {e}")
            import traceback
            traceback.print_exc()
        
        # 对预测结果进行合理性检查和安全约束
        last_price = data['单位净值'].iloc[-1]
        logger.info(f"\n=== 预测结果合理性检查 ===")
        logger.info(f"昨日实际净值: {last_price:.4f}")
        
        # 获取所有模型的预测结果
        all_predictions = {
            "最优模型": {
                "predictions": multi_step_predictions,
                "model_name": predictor.current_best_model
            }
        }
        
        # 添加其他模型的预测结果
        for model_name, preds in predictor.predictions.items():
            if model_name not in all_predictions and isinstance(preds, list):
                all_predictions[model_name] = {
                    "predictions": preds,
                    "model_name": model_name
                }
        
        # 对每个模型的预测结果进行检查和约束
        for display_name, model_info in all_predictions.items():
            predictions = model_info["predictions"]
            model_name = model_info["model_name"]
            
            # 1. 进行合理性检查
            check_result = sanity_check(predictions, last_price, data, model_name)
            
            # 2. 应用安全约束
            constrained_preds = apply_safety_constraints(predictions, last_price, data)
            
            # 3. 更新预测结果
            if display_name == "最优模型":
                multi_step_predictions = constrained_preds
                # 更新predictor.predictions中的结果
                if model_name in predictor.predictions:
                    predictor.predictions[model_name] = constrained_preds
            elif model_name in predictor.predictions:
                predictor.predictions[model_name] = constrained_preds
            
            # 4. 输出检查结果
            logger.info(f"\n{display_name} ({model_name}) 检查结果:")
            for check in check_result["checks"]:
                logger.info(f"  {check}")
            for warning in check_result["warnings"]:
                logger.warning(f"  ⚠️ {warning}")
            for error in check_result["errors"]:
                logger.error(f"  ❌ {error}")
            
            if check_result["warnings"] or check_result["errors"]:
                logger.info(f"  应用安全约束后预测结果:")
                for i in range(min(len(constrained_preds), forecast_steps)):
                    logger.info(f"  第{i+1}天: {constrained_preds[i]:.4f}")
        
        logger.info(f"\n=== 最终预测结果 ===")
        logger.info(f"当前使用模型: {predictor.current_best_model}")
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
        
        # 打印Transformer模型预测结果（如果有）
        transformer_predictions = predictor.predictions.get('Transformer', None)
        if transformer_predictions is not None:
            logger.info("\nTransformer模型预测结果:")
            for i in range(forecast_steps):
                logger.info(f"第{i+1}天: {transformer_predictions[i]:.4f}")
        
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
        
        # 初始化风险管理模块
        logger.info("\n=== 初始化风险管理模块 ===")
        risk_manager = RiskManagement(initial_capital=100000, risk_free_rate=0.025)
        
        # 初始化压力测试模块
        logger.info("\n=== 初始化压力测试模块 ===")
        stress_tester = StressTesting(risk_manager)
        
        # 准备回测数据
        logger.info("\n=== 准备回测数据 ===")
        backtest_data = data.copy()
        backtest_data = backtest_data.reset_index()
        backtest_data = backtest_data.rename(columns={
            '净值日期': 'date',
            '单位净值': 'close'
        })
        
        # 初始化回测系统，优化参数
        logger.info("\n=== 初始化回测系统 ===")
        backtesting_system = BacktestingSystem(
            initial_capital=100000,
            transaction_cost=0.0005,  # 降低交易成本
            slippage=0.0002,  # 降低滑点
            commission=0.0001,  # 降低佣金
            market_impact=0.0002  # 降低市场冲击
        )
        
        # 定义回测策略函数
        def backtest_strategy(historical_data, forecast_steps=1):
            # 简化的策略：基于简单移动平均线交叉
            if len(historical_data) < 10:
                return 0, historical_data['close'].iloc[-1]  # 持有信号
            
            # 使用较短的移动平均线窗口，增加交易信号
            short_ma = historical_data['close'].rolling(window=3).mean().iloc[-1]  # 短期均线
            long_ma = historical_data['close'].rolling(window=7).mean().iloc[-1]  # 长期均线
            
            # 简单的移动平均线交叉策略
            if short_ma > long_ma:
                return 1, historical_data['close'].iloc[-1] * 1.01  # 买入信号
            elif short_ma < long_ma:
                return -1, historical_data['close'].iloc[-1] * 0.99  # 卖出信号
            else:
                return 0, historical_data['close'].iloc[-1]  # 持有信号
        
        # 运行回测
        logger.info("\n=== 开始回测 ===")
        backtest_results = backtesting_system.run_backtest(
            data=backtest_data,
            strategy_func=backtest_strategy,
            lookback=20,
            forecast_steps=1
        )
        
        # 生成回测报告
        logger.info("\n=== 生成回测报告 ===")
        backtest_report = backtesting_system.generate_backtest_report()
        backtesting_system.plot_backtest_results()
        
        # 检查模型漂移和性能衰减
        logger.info("\n=== 检查模型漂移和性能衰减 ===")
        model_performance = backtesting_system.analyze_model_performance()
        
        # 检查结果中是否包含所需的键
        model_drift = model_performance.get('model_drift', False)
        performance_decay = model_performance.get('performance_decay', False)
        
        if model_drift:
            logger.warning("检测到模型漂移，建议重新训练模型")
        else:
            logger.info("未检测到模型漂移，模型表现稳定")
        
        if performance_decay:
            logger.warning("检测到性能衰减，建议调整模型参数")
        else:
            logger.info("未检测到性能衰减，模型性能稳定")
        
        # 计算风险指标 - 使用回测结果而非历史数据
        logger.info("\n=== 计算风险指标 ===")
        
        # 从回测结果中提取收益序列
        backtest_returns = []
        if len(backtesting_system.equity_curve) > 1:
            for i in range(1, len(backtesting_system.equity_curve)):
                prev_equity = backtesting_system.equity_curve[i-1]['equity']
                curr_equity = backtesting_system.equity_curve[i]['equity']
                daily_return = (curr_equity - prev_equity) / prev_equity
                backtest_returns.append(daily_return)
        
        if backtest_returns:
            backtest_returns = np.array(backtest_returns)
            risk_metrics = risk_manager.calculate_risk_metrics(backtest_returns)
            
            # 风险评估
            logger.info("\n风险评估指标 (基于回测结果):")
            logger.info(f"日波动率: {risk_metrics['return_std']*100:.4f}%")
            logger.info(f"年化波动率: {risk_metrics['annual_std']*100:.4f}%")
            logger.info(f"年化平均收益率: {risk_metrics['annual_return']*100:.4f}%")
            logger.info(f"夏普比率: {risk_metrics['sharpe_ratio']:.4f}")
            logger.info(f"最大回撤: {risk_metrics['max_drawdown']*100:.4f}%")
            logger.info(f"Sortino比率: {risk_metrics['sortino_ratio']:.4f}")
            logger.info(f"Value at Risk (95%): {risk_metrics['value_at_risk_95']*100:.4f}%")
            logger.info(f"Conditional Value at Risk (95%): {risk_metrics['conditional_var_95']*100:.4f}%")
            logger.info(f"胜率: {risk_metrics['win_rate']*100:.2f}%")
            logger.info(f"盈利因子: {risk_metrics['profit_factor']:.4f}")
        else:
            # 如果没有回测收益，使用历史数据
            returns = data['simple_return'].dropna() / 100  # 转换为小数形式
            risk_metrics = risk_manager.calculate_risk_metrics(returns)
            
            # 风险评估
            logger.info("\n风险评估指标 (基于历史数据):")
            logger.info(f"日波动率: {risk_metrics['return_std']*100:.4f}%")
            logger.info(f"年化波动率: {risk_metrics['annual_std']*100:.4f}%")
            logger.info(f"年化平均收益率: {risk_metrics['annual_return']*100:.4f}%")
            logger.info(f"夏普比率: {risk_metrics['sharpe_ratio']:.4f}")
            logger.info(f"最大回撤: {risk_metrics['max_drawdown']*100:.4f}%")
            logger.info(f"Sortino比率: {risk_metrics['sortino_ratio']:.4f}")
            logger.info(f"Value at Risk (95%): {risk_metrics['value_at_risk_95']*100:.4f}%")
            logger.info(f"Conditional Value at Risk (95%): {risk_metrics['conditional_var_95']*100:.4f}%")
            logger.info(f"胜率: {risk_metrics['win_rate']*100:.2f}%")
            logger.info(f"盈利因子: {risk_metrics['profit_factor']:.4f}")
        
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
        
        # 运行市场结构变化检测，实现适应性进化
        logger.info("\n=== 运行市场结构变化检测 ===")
        
        # 准备数据
        price_data = data['单位净值']
        
        # 检测市场结构变化
        market_structure_result = stress_tester.market_structure_change_test(price_data)
        
        # 基于市场结构变化调整模型参数，实现适应性进化
        logger.info("\n=== 实现适应性进化 ===")
        if market_structure_result['structure_change_detected']:
            logger.warning("检测到市场结构变化，开始调整模型参数...")
            
            # 根据波动率变化调整模型参数
            volatility_change = market_structure_result['volatility_increase']
            
            # 示例：如果波动率增加超过20%，调整回测参数
            if volatility_change > 0.2:
                logger.info(f"波动率增加了{volatility_change:.2%}，调整回测参数...")
                # 这里可以添加具体的参数调整逻辑
                # 例如：降低交易频率，增加止损比例等
                backtesting_system.transaction_cost = 0.001  # 提高交易成本以反映市场变化
                backtesting_system.slippage = 0.0005  # 提高滑点以反映市场变化
                
                logger.info("模型参数调整完成，已适应新的市场结构")
        else:
            logger.info("未检测到市场结构变化，模型参数保持不变")
        
        # 将市场结构变化结果保存到压力测试结果中
        stress_test_results = {
            'market_structure_test': market_structure_result
        }
        
        # 天龙人模型：检测和预测极端价格变动
        logger.info("\n=== 运行天龙人模型 ===")
        
        class CelestialDragonModel:
            """天龙人模型 - 检测和预测极端价格变动"""
            
            def __init__(self, data, threshold=2):
                """初始化天龙人模型
                
                参数:
                data: 价格数据
                threshold: 极端变动的标准差阈值
                """
                self.data = data
                self.threshold = threshold
                self.returns = data.pct_change().dropna()
                self.mean_return = self.returns.mean()
                self.std_return = self.returns.std()
            
            def detect_extreme_events(self):
                """检测历史极端价格变动事件"""
                # 计算极端变动阈值
                upper_threshold = self.mean_return + self.threshold * self.std_return
                lower_threshold = self.mean_return - self.threshold * self.std_return
                
                # 检测极端事件
                extreme_events = {
                    'upper': self.returns[self.returns > upper_threshold],
                    'lower': self.returns[self.returns < lower_threshold]
                }
                
                return extreme_events, upper_threshold, lower_threshold
            
            def analyze_extreme_events(self, extreme_events):
                """分析极端价格变动事件的特征和规律"""
                # 计算极端事件的统计特征
                analysis = {
                    'total_extreme_events': len(extreme_events['upper']) + len(extreme_events['lower']),
                    'up_events_count': len(extreme_events['upper']),
                    'down_events_count': len(extreme_events['lower']),
                    'avg_up_magnitude': extreme_events['upper'].mean() * 100 if len(extreme_events['upper']) > 0 else 0,
                    'avg_down_magnitude': extreme_events['lower'].mean() * 100 if len(extreme_events['lower']) > 0 else 0,
                    'max_up_magnitude': extreme_events['upper'].max() * 100 if len(extreme_events['upper']) > 0 else 0,
                    'max_down_magnitude': extreme_events['lower'].min() * 100 if len(extreme_events['lower']) > 0 else 0
                }
                
                return analysis
            
            def predict_extreme_change(self):
                """预测未来是否会发生极端价格变动"""
                # 计算最近的波动率
                recent_returns = self.returns[-30:]  # 最近30天的收益率
                recent_volatility = recent_returns.std()
                
                # 计算最近的平均收益率
                recent_mean = recent_returns.mean()
                
                # 计算最近的最大和最小收益率
                recent_max = recent_returns.max()
                recent_min = recent_returns.min()
                
                # 预测逻辑：如果最近波动率上升，且最近有接近极端阈值的变动，则预测可能发生极端变动
                prediction = {
                    'extreme_change_probability': 0,
                    'direction': 'none',
                    'confidence': 0
                }
                
                # 计算波动率变化率
                volatility_change = (recent_volatility - self.std_return) / self.std_return
                
                # 如果波动率增加超过30%，则认为风险增加
                if volatility_change > 0.3:
                    # 检查最近是否有接近极端阈值的变动
                    upper_near_threshold = recent_max > self.mean_return + 1.5 * recent_volatility
                    lower_near_threshold = recent_min < self.mean_return - 1.5 * recent_volatility
                    
                    if upper_near_threshold or lower_near_threshold:
                        # 计算极端变动的概率
                        prediction['extreme_change_probability'] = 0.7 if upper_near_threshold or lower_near_threshold else 0.3
                        
                        # 预测方向
                        if upper_near_threshold:
                            prediction['direction'] = 'up'
                        elif lower_near_threshold:
                            prediction['direction'] = 'down'
                        
                        # 计算置信度
                        prediction['confidence'] = min(1.0, abs(recent_max - recent_min) / self.std_return)
                
                return prediction
        
        # 初始化天龙人模型
        celestial_dragon = CelestialDragonModel(data['单位净值'])
        
        # 检测极端事件
        extreme_events, upper_threshold, lower_threshold = celestial_dragon.detect_extreme_events()
        
        # 分析极端事件
        extreme_analysis = celestial_dragon.analyze_extreme_events(extreme_events)
        
        logger.info(f"极端变动阈值：上涨 {upper_threshold*100:.2f}%，下跌 {lower_threshold*100:.2f}%")
        logger.info(f"历史极端事件总数：{extreme_analysis['total_extreme_events']}次")
        logger.info(f"上涨极端事件：{extreme_analysis['up_events_count']}次，平均幅度：{extreme_analysis['avg_up_magnitude']:.2f}%")
        logger.info(f"下跌极端事件：{extreme_analysis['down_events_count']}次，平均幅度：{extreme_analysis['avg_down_magnitude']:.2f}%")
        logger.info(f"最大上涨幅度：{extreme_analysis['max_up_magnitude']:.2f}%")
        logger.info(f"最大下跌幅度：{extreme_analysis['max_down_magnitude']:.2f}%")
        
        # 预测极端变动
        extreme_prediction = celestial_dragon.predict_extreme_change()
        
        logger.info(f"极端变动预测概率：{extreme_prediction['extreme_change_probability']:.2f}")
        logger.info(f"预测方向：{extreme_prediction['direction']}")
        logger.info(f"预测置信度：{extreme_prediction['confidence']:.2f}")
        
        # 保存天龙人模型结果
        celestial_dragon_result = {
            'extreme_events': extreme_events,
            'extreme_analysis': extreme_analysis,
            'extreme_prediction': extreme_prediction,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold
        }
        
        # 交易信号生成
        today_prediction = multi_step_predictions[0]
        today_change = daily_changes[0]
        
        # 基于预测和风险指标生成初始交易建议
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
        
        # 天龙人模型结果作为强条件，调整交易信号
        logger.info("\n=== 应用天龙人模型强条件 ===")
        
        # 获取天龙人模型的预测结果
        extreme_prob = extreme_prediction['extreme_change_probability']
        extreme_direction = extreme_prediction['direction']
        extreme_confidence = extreme_prediction['confidence']
        
        # 如果天龙人模型预测有高概率的极端变动，将其作为强条件
        if extreme_prob > 0.5 and extreme_confidence > 0.7:
            logger.info(f"天龙人模型预测极端变动概率高 ({extreme_prob:.2f})，置信度高 ({extreme_confidence:.2f})")
            logger.info(f"预测方向：{extreme_direction}")
            
            # 根据极端变动方向调整交易信号
            if extreme_direction == 'up':
                signal = "强烈买入"
                confidence = "高"
                logger.info("应用强条件：上调交易信号为'强烈买入'")
            elif extreme_direction == 'down':
                signal = "强烈卖出"
                confidence = "高"
                logger.info("应用强条件：上调交易信号为'强烈卖出'")
        
        logger.info(f"\n最终交易信号:")
        logger.info(f"今日预测涨跌幅: {today_change:.2f}%")
        logger.info(f"建议操作: {signal}")
        logger.info(f"信号置信度: {confidence}")
        
        # 计算概率性预测结果
        logger.info("\n=== 计算概率性预测结果 ===")
        probabilistic_results = None
        try:
            # 使用最优模型进行概率性预测
            probabilistic_results = predictor.probabilistic_forecast(
                model_name=best_model,
                forecast_steps=forecast_steps,
                n_simulations=1000,
                feature_matrix=feature_matrix if 'feature_matrix' in locals() else None
            )
            
            logger.info("概率性预测结果：")
            logger.info(f"  中位数预测: {probabilistic_results['median']}")
            logger.info(f"  均值预测: {probabilistic_results['mean']}")
            logger.info(f"  80%置信区间: [{probabilistic_results['lower_80']}, {probabilistic_results['upper_80']}]")
            logger.info(f"  上涨概率: [{p*100:.1f}% for p in probabilistic_results['probability_up']]")
        except Exception as e:
            logger.warning(f"概率性预测失败：{e}")
            import traceback
            traceback.print_exc()
        
        # 生成交易信号和止损止盈建议
        logger.info("\n=== 生成交易信号和止损止盈建议 ===")
        advisor = GoldTradingAdvisor()
        
        # 准备预测结果和不确定性指标
        prediction_data = {
            'mean': multi_step_predictions
        }
        
        uncertainty_data = {
            'probability_up': probabilistic_results['probability_up'] if probabilistic_results else [0.5],
            'expected_max_loss': probabilistic_results['expected_max_loss'][0] if probabilistic_results else 0
        }
        
        # 生成交易建议
        trading_signal = advisor.generate_signal(
            prediction=prediction_data,
            uncertainty_metrics=uncertainty_data,
            current_price=last_price,
            hist_data=data
        )
        
        # 输出交易建议
        logger.info(f"\n交易建议:")
        logger.info(f"  信号: {trading_signal['signal']}")
        logger.info(f"  置信度: {trading_signal['confidence']}")
        logger.info(f"  建议仓位: {trading_signal['position_size']}")
        logger.info(f"  预期收益率: {trading_signal['expected_return']}")
        logger.info(f"  上涨概率: {trading_signal['probability_up']}")
        logger.info(f"  止损价格: {trading_signal['stop_loss']}")
        logger.info(f"  止盈价格: {trading_signal['take_profit']}")
        logger.info(f"  风险收益比: {trading_signal['risk_reward_ratio']}")
        logger.info(f"  日波动率: {trading_signal['daily_volatility']}")
        logger.info(f"  年化波动率: {trading_signal['annual_volatility']}")
        
        # 更新交易信号
        signal = trading_signal['signal']
        confidence = trading_signal['confidence']
        
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
            'probabilistic_results': probabilistic_results,
            'trading_signal': trading_signal,
            'data_sample_size': len(data),
            'data_date_range': {
                'start': data.index.min(),
                'end': data.index.max()
            },
            'historical_stats': {
                'avg_daily_return': data['log_return'].mean(),
                'std_daily_return': data['log_return'].std(),
                'max_daily_gain': data['simple_return'].max(),
                'max_daily_loss': data['simple_return'].min()
            },
            'risk_assessment': {
                'volatility': risk_metrics['annual_std'] * 100,
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown'] * 100,
                'sortino_ratio': risk_metrics['sortino_ratio'],
                'value_at_risk_95': risk_metrics['value_at_risk_95'] * 100,
                'conditional_var_95': risk_metrics['conditional_var_95'] * 100,
                'win_rate': risk_metrics['win_rate'] * 100,
                'profit_factor': risk_metrics['profit_factor']
            },
            'trading_signal': {
                'signal': signal,
                'confidence': confidence,
                'reason': f"基于{best_model}模型预测，今日预计{'上涨' if today_change > 0 else '下跌'}{abs(today_change):.2f}%"
            },
            'backtesting_results': backtest_results,
            'backtesting_report': backtest_report,
            'model_performance': model_performance,
            'model_drift': model_performance['model_drift'],
            'performance_decay': model_performance['performance_decay'],
            'stress_test_results': stress_test_results,
            'celestial_dragon_results': {
                'extreme_analysis': extreme_analysis,
                'extreme_prediction': extreme_prediction,
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold
            }
        }
        
        # 添加高级模型预测结果（如果有）
        if xgb_predictions is not None:
            result['xgb_predictions'] = xgb_predictions
        
        if lstm_predictions is not None:
            result['lstm_predictions'] = lstm_predictions
        
        # 添加Transformer模型预测结果（如果有）
        transformer_predictions = predictor.predictions.get('Transformer', None)
        if transformer_predictions is not None:
            result['transformer_predictions'] = transformer_predictions
            
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
    
    # 打印Transformer模型预测结果（如果有）
    if 'transformer_predictions' in result:
        print("\n【Transformer模型预测】")
        for i in range(len(result['transformer_predictions'])):
            print(f"第{i+1}天：{result['transformer_predictions'][i]:.4f}")
    
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
    print(f"Sortino比率：{result['risk_assessment']['sortino_ratio']:.4f}")
    print(f"最大回撤：{result['risk_assessment']['max_drawdown']:.4f}%")
    print(f"Value at Risk (95%)：{result['risk_assessment']['value_at_risk_95']:.4f}%")
    print(f"Conditional Value at Risk (95%)：{result['risk_assessment']['conditional_var_95']:.4f}%")
    print(f"历史胜率：{result['risk_assessment']['win_rate']:.2f}%")
    print(f"盈利因子：{result['risk_assessment']['profit_factor']:.4f}")
    
    print("\n【回测结果】")
    print(f"回测起始日期：{result['backtesting_report']['summary']['start_date']}")
    print(f"回测结束日期：{result['backtesting_report']['summary']['end_date']}")
    print(f"回测总天数：{result['backtesting_report']['summary']['total_days']}")
    print(f"回测总交易次数：{result['backtesting_report']['summary']['total_trades']}")
    print(f"回测总收益：{result['backtesting_report']['summary']['total_return']*100:.2f}%")
    print(f"回测年化收益：{result['backtesting_report']['performance_metrics']['annualized_return']*100:.2f}%")
    print(f"回测夏普比率：{result['backtesting_report']['performance_metrics']['sharpe_ratio']:.4f}")
    print(f"回测最大回撤：{result['backtesting_report']['performance_metrics']['max_drawdown']*100:.4f}%")
    
    print("\n【模型稳定性】")
    print(f"模型漂移检测：{'检测到模型漂移' if result['model_drift'] else '未检测到模型漂移'}")
    print(f"性能衰减检测：{'检测到性能衰减' if result['performance_decay'] else '未检测到性能衰减'}")
    
    print("\n【交易建议】")
    print(f"建议操作：{result['trading_signal']['signal']}")
    print(f"信号置信度：{result['trading_signal']['confidence']}")
    print(f"建议理由：{result['trading_signal']['reason']}")
    
    print("\n" + "="*70)
    print("⚠️  风险提示：本预测基于历史数据的统计模型，不构成投资建议")
    print("⚠️  投资有风险，入市需谨慎")
    print("="*70)
