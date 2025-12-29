"""
苏格拉底时间序列预测系统测试脚本
"""

import pandas as pd
import numpy as np
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor

print("=" * 60)
print("苏格拉底时间序列预测系统 - 测试开始")
print("=" * 60)

# 1. 测试数据处理模块
try:
    print("\n1. 测试数据处理模块...")
    processor = DataProcessor()
    
    # 加载数据
    data = processor.load_data(
        source='test_data.csv',
        time_column='date',
        target_column='value',
        source_type='file',
        file_format='csv'
    )
    print("   ✓ 数据加载成功")
    print(f"   数据量: {len(data)} 条记录")
    print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
    
    # 清洗数据
    cleaned_data = processor.clean_data(drop_na=True, method='ffill', threshold=3)
    print("   ✓ 数据清洗成功")
    
    # 检验平稳性
    is_stationary = processor.check_stationarity()
    print(f"   序列平稳性: {'平稳' if is_stationary else '非平稳'}")
    
    # 划分训练集和测试集
    train_data, test_data = processor.split_data(train_size=0.8)
    print(f"   ✓ 数据划分成功: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
    
except Exception as e:
    print(f"   ✗ 数据处理模块测试失败: {str(e)}")

# 2. 测试预测模块
try:
    print("\n2. 测试预测模块...")
    # 重新加载数据以确保data变量存在
    if 'data' not in locals() or data is None:
        processor = DataProcessor()
        data = processor.load_data(
            source='test_data.csv',
            time_column='date',
            target_column='value',
            source_type='file',
            file_format='csv'
        )
    predictor = TimeSeriesPredictor(data, processor.target_column)
    
    # 测试移动平均
    ma_forecasts = predictor.moving_average(window_size=3, forecast_steps=3)
    print(f"   ✓ 移动平均预测成功: {[round(x, 2) for x in ma_forecasts]}")
    
    # 测试指数平滑
    es_forecasts = predictor.exponential_smoothing(alpha=0.2, forecast_steps=3)
    print(f"   ✓ 指数平滑预测成功: {[round(x, 2) for x in es_forecasts]}")
    
    # 测试ARIMA
    arima_forecasts = predictor.arima(order=(1, 1, 1), forecast_steps=3)
    print(f"   ✓ ARIMA预测成功: {[round(x, 2) for x in arima_forecasts]}")
    
    # 测试Holt-Winters
    hw_forecasts = predictor.holt_winters(seasonal_periods=12, forecast_steps=3)
    print(f"   ✓ Holt-Winters预测成功: {[round(x, 2) for x in hw_forecasts]}")
    
    # 评估模型
    evaluations = predictor.evaluate(test_data)
    print("   ✓ 模型评估成功")
    print("   模型性能指标:")
    for model, metrics in evaluations.items():
        print(f"      {model}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    
    # 选择最优模型
    best_model = predictor.get_best_model()
    print(f"   ✓ 最优模型选择: {best_model}")
    
    # 使用最优模型预测
    best_forecasts = predictor.predict_with_best_model(forecast_steps=3)
    print(f"   ✓ 最优模型预测: {[round(x, 2) for x in best_forecasts]}")
    
except Exception as e:
    print(f"   ✗ 预测模块测试失败: {str(e)}")

# 3. 测试系统集成
try:
    print("\n3. 测试系统集成...")
    
    # 完整工作流测试
    print("   执行完整工作流测试...")
    
    # 1. 加载和预处理
    processor = DataProcessor()
    data = processor.load_data('test_data.csv', 'date', 'value')
    cleaned_data = processor.clean_data()
    
    # 2. 创建预测器
    predictor = TimeSeriesPredictor(cleaned_data, processor.target_column)
    
    # 3. 训练所有模型
    predictor.moving_average(window_size=3, forecast_steps=1)
    predictor.exponential_smoothing(alpha=0.2, forecast_steps=1)
    predictor.arima(order=(1, 1, 1), forecast_steps=1)
    predictor.holt_winters(seasonal_periods=12, forecast_steps=1)
    
    # 4. 评估并选择最优模型
    evaluations = predictor.evaluate()
    best_model = predictor.get_best_model()
    
    # 5. 进行多步预测
    long_term_forecasts = predictor.predict_with_best_model(forecast_steps=6)
    
    print(f"   ✓ 完整工作流测试成功")
    print(f"   最优模型: {best_model}")
    print(f"   未来6个月预测: {[round(x, 2) for x in long_term_forecasts]}")
    
except Exception as e:
    print(f"   ✗ 系统集成测试失败: {str(e)}")

# 4. 创建requirements.txt
try:
    print("\n4. 创建依赖文件...")
    requirements = """
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
python-multipart>=0.0.5
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("   ✓ requirements.txt 创建成功")
    
except Exception as e:
    print(f"   ✗ 依赖文件创建失败: {str(e)}")

# 5. 创建README.md
try:
    print("\n5. 创建README文档...")
    readme = """
# 苏格拉底时间序列预测系统

基于时间序列分析的未来预测系统，能够通过历史数据预测未来趋势和事件。

## 功能特性

### 数据处理
- 支持多种格式数据导入 (CSV, Excel, JSON)
- 智能数据清洗 (缺失值、异常值处理)
- 数据预处理 (归一化、差分、平稳性检验)
- 数据可视化 (时间序列图、趋势图、季节性图)

### 预测算法
- 移动平均 (MA)
- 指数平滑 (ES)
- ARIMA
- Holt-Winters (支持季节性)

### 用户界面
- 命令行界面 (CLI)
- RESTful API (基于FastAPI)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 使用命令行界面

1. **加载数据**
```bash
python cli.py load --file test_data.csv --time-col date --target-col value
```

2. **清洗数据**
```bash
python cli.py clean --drop-na --method ffill
```

3. **进行预测**
```bash
python cli.py predict --algorithm best --steps 3
```

### 使用Web API

1. **启动API服务**
```bash
python api.py
```

2. **访问API文档**
```
http://localhost:8000/docs
```

## 示例

```python
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor

# 加载数据
processor = DataProcessor()
data = processor.load_data('test_data.csv', 'date', 'value')

# 创建预测器
predictor = TimeSeriesPredictor(data, 'value')

# 预测未来3个值
forecasts = predictor.predict_with_best_model(forecast_steps=3)
print(forecasts)
```

## 项目结构

```
苏格拉底/
├── data_processing.py    # 数据处理模块
├── prediction.py         # 预测算法模块
├── cli.py                # 命令行界面
├── api.py                # Web API接口
├── test_data.csv         # 测试数据
├── test_system.py        # 系统测试脚本
├── requirements.txt      # 依赖文件
└── README.md             # 项目文档
```

## 许可证

MIT License
"""
    
    with open('README.md', 'w') as f:
        f.write(readme.strip())
    
    print("   ✓ README.md 创建成功")
    
except Exception as e:
    print(f"   ✗ 文档创建失败: {str(e)}")

print("\n" + "=" * 60)
print("苏格拉底时间序列预测系统 - 测试完成")
print("=" * 60)
