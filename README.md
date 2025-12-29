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