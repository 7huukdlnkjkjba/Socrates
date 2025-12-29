from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import io
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
import json

app = FastAPI(
    title="苏格拉底时间序列预测系统 API",
    description="基于时间序列分析的未来预测系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储全局处理器实例
processors = {}
predictors = {}

def get_processor(request_id: str):
    if request_id not in processors:
        raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
    return processors[request_id]

def get_predictor(request_id: str):
    if request_id not in predictors:
        raise HTTPException(status_code=404, detail=f"Predictor for request ID {request_id} not found")
    return predictors[request_id]

@app.post("/api/v1/load-data/{request_id}")
async def load_data(
    request_id: str,
    file: UploadFile = File(...),
    time_col: str = "date",
    target_col: str = "value",
    file_format: str = "csv"
):
    """加载时间序列数据"""
    try:
        # 读取文件内容
        content = await file.read()
        
        # 创建数据处理器
        processor = DataProcessor()
        
        # 根据文件格式加载数据
        if file_format == "csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_format == "excel":
            df = pd.read_excel(io.BytesIO(content))
        elif file_format == "json":
            df = pd.read_json(io.StringIO(content.decode("utf-8")))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_format}")
        
        # 设置时间列和目标列
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        # 存储处理器实例
        processor.data = df
        processor.time_column = time_col
        processor.target_column = target_col
        processors[request_id] = processor
        
        return {
            "success": True,
            "message": "数据加载成功",
            "request_id": request_id,
            "data_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "time_range": {
                    "start": str(df.index.min()),
                    "end": str(df.index.max())
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/clean-data/{request_id}")
async def clean_data(
    request_id: str,
    drop_na: bool = True,
    method: str = "ffill",
    threshold: int = 3
):
    """清洗数据"""
    processor = get_processor(request_id)
    
    try:
        processor.clean_data(drop_na=drop_na, method=method, threshold=threshold)
        return {
            "success": True,
            "message": "数据清洗成功",
            "request_id": request_id,
            "cleaned_rows": len(processor.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/preprocess-data/{request_id}")
async def preprocess_data(
    request_id: str,
    make_stationary: bool = False,
    method: str = "diff",
    order: int = 1,
    normalize: bool = False,
    norm_method: str = "min-max"
):
    """预处理数据"""
    processor = get_processor(request_id)
    
    try:
        if make_stationary:
            processor.make_stationary(method=method, order=order)
        
        if normalize:
            processor.normalize(method=norm_method)
        
        return {
            "success": True,
            "message": "数据预处理成功",
            "request_id": request_id,
            "target_column": processor.target_column
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/{request_id}")
async def predict(
    request_id: str,
    algorithm: str,
    steps: int = 1,
    window_size: int = 3,
    alpha: float = 0.2,
    order: list = [1, 1, 1],
    seasonal: int = 12
):
    """进行时间序列预测"""
    processor = get_processor(request_id)
    
    try:
        # 创建预测器
        predictor = TimeSeriesPredictor(processor.get_data(), processor.target_column)
        predictors[request_id] = predictor
        
        # 执行预测
        if algorithm == "MA":
            forecasts = predictor.moving_average(window_size=window_size, forecast_steps=steps)
        elif algorithm == "ES":
            forecasts = predictor.exponential_smoothing(alpha=alpha, forecast_steps=steps)
        elif algorithm == "ARIMA":
            forecasts = predictor.arima(order=tuple(order), forecast_steps=steps)
        elif algorithm == "Holt-Winters":
            forecasts = predictor.holt_winters(seasonal_periods=seasonal, forecast_steps=steps)
        elif algorithm == "best":
            forecasts = predictor.predict_with_best_model(forecast_steps=steps)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")
        
        return {
            "success": True,
            "message": "预测完成",
            "request_id": request_id,
            "algorithm": algorithm,
            "forecast_steps": steps,
            "forecasts": forecasts,
            "target_column": processor.target_column
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/evaluate/{request_id}")
async def evaluate(request_id: str):
    """评估模型性能"""
    predictor = get_predictor(request_id)
    
    try:
        evaluations = predictor.evaluate()
        best_model = predictor.get_best_model()
        
        return {
            "success": True,
            "message": "模型评估完成",
            "request_id": request_id,
            "evaluations": evaluations,
            "best_model": best_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data-info/{request_id}")
async def data_info(request_id: str):
    """获取数据信息"""
    processor = get_processor(request_id)
    
    try:
        df = processor.get_data()
        return {
            "success": True,
            "message": "获取数据信息成功",
            "request_id": request_id,
            "data_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "time_range": {
                    "start": str(df.index.min()),
                    "end": str(df.index.max())
                },
                "target_column": processor.target_column,
                "statistics": {
                    "mean": float(df[processor.target_column].mean()),
                    "std": float(df[processor.target_column].std()),
                    "min": float(df[processor.target_column].min()),
                    "max": float(df[processor.target_column].max()),
                    "median": float(df[processor.target_column].median())
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/clear/{request_id}")
async def clear_request(request_id: str):
    """清除请求数据"""
    try:
        if request_id in processors:
            del processors[request_id]
        if request_id in predictors:
            del predictors[request_id]
        
        return {
            "success": True,
            "message": "请求数据已清除",
            "request_id": request_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用苏格拉底时间序列预测系统！",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
