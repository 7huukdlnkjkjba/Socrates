import argparse
import sys
from data_processing import DataProcessor
from prediction import TimeSeriesPredictor
import pandas as pd

class SocratesCLI:
    def __init__(self):
        self.processor = None
        self.predictor = None
        self.args = None
    
    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="苏格拉底时间序列预测系统 - 预知未来！")
        
        # 主命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 加载数据命令
        load_parser = subparsers.add_parser('load', help='加载时间序列数据')
        load_parser.add_argument('--file', required=True, help='数据文件路径')
        load_parser.add_argument('--time-col', required=True, help='时间列名称')
        load_parser.add_argument('--target-col', required=True, help='目标列名称')
        load_parser.add_argument('--format', default='csv', choices=['csv', 'excel', 'json'], help='文件格式')
        
        # 清洗数据命令
        clean_parser = subparsers.add_parser('clean', help='清洗数据')
        clean_parser.add_argument('--drop-na', action='store_true', help='删除缺失值')
        clean_parser.add_argument('--method', default='ffill', choices=['ffill', 'bfill', 'mean', 'median'], help='填充方法')
        clean_parser.add_argument('--threshold', type=int, default=3, help='异常值Z-score阈值')
        
        # 预处理命令
        preprocess_parser = subparsers.add_parser('preprocess', help='预处理数据')
        preprocess_parser.add_argument('--make-stationary', action='store_true', help='使序列平稳化')
        preprocess_parser.add_argument('--method', default='diff', choices=['diff', 'log', 'sqrt'], help='平稳化方法')
        preprocess_parser.add_argument('--order', type=int, default=1, help='差分阶数')
        preprocess_parser.add_argument('--normalize', action='store_true', help='数据归一化')
        preprocess_parser.add_argument('--norm-method', default='min-max', choices=['min-max', 'z-score'], help='归一化方法')
        
        # 预测命令
        predict_parser = subparsers.add_parser('predict', help='进行时间序列预测')
        predict_parser.add_argument('--algorithm', required=True, choices=['MA', 'ES', 'ARIMA', 'Holt-Winters', 'best'], help='预测算法')
        predict_parser.add_argument('--steps', type=int, default=1, help='预测步数')
        predict_parser.add_argument('--window-size', type=int, default=3, help='移动平均窗口大小')
        predict_parser.add_argument('--alpha', type=float, default=0.2, help='指数平滑参数')
        predict_parser.add_argument('--order', nargs=3, type=int, default=[1, 1, 1], help='ARIMA模型阶数 (p,d,q)')
        predict_parser.add_argument('--seasonal', type=int, default=12, help='季节性周期')
        
        # 可视化命令
        visualize_parser = subparsers.add_parser('visualize', help='可视化数据')
        visualize_parser.add_argument('--type', required=True, choices=['ts', 'trend', 'seasonality', 'histogram'], help='可视化类型')
        
        # 评估命令
        evaluate_parser = subparsers.add_parser('evaluate', help='评估模型性能')
        
        # 显示数据信息命令
        info_parser = subparsers.add_parser('info', help='显示数据信息')
        
        self.args = parser.parse_args()
        
        if not self.args.command:
            parser.print_help()
            sys.exit(1)
    
    def run(self):
        """执行命令"""
        self.parse_args()
        
        try:
            if self.args.command == 'load':
                self.load_data()
            elif self.args.command == 'clean':
                self.clean_data()
            elif self.args.command == 'preprocess':
                self.preprocess_data()
            elif self.args.command == 'predict':
                self.predict()
            elif self.args.command == 'visualize':
                self.visualize()
            elif self.args.command == 'evaluate':
                self.evaluate()
            elif self.args.command == 'info':
                self.show_info()
        except Exception as e:
            print(f"错误: {str(e)}")
            sys.exit(1)
    
    def load_data(self):
        """加载数据"""
        print(f"正在加载数据...")
        self.processor = DataProcessor()
        data = self.processor.load_data(
            file_path=self.args.file,
            time_column=self.args.time_col,
            target_column=self.args.target_col,
            file_format=self.args.format
        )
        print(f"数据加载成功! 共 {len(data)} 条记录")
        print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
    
    def clean_data(self):
        """清洗数据"""
        if not self.processor:
            raise ValueError("请先加载数据")
        
        print(f"正在清洗数据...")
        data = self.processor.clean_data(
            drop_na=self.args.drop_na,
            method=self.args.method,
            threshold=self.args.threshold
        )
        print(f"数据清洗成功! 剩余 {len(data)} 条记录")
    
    def preprocess_data(self):
        """预处理数据"""
        if not self.processor:
            raise ValueError("请先加载数据")
        
        print(f"正在预处理数据...")
        
        if self.args.make_stationary:
            data = self.processor.make_stationary(
                method=self.args.method,
                order=self.args.order
            )
            print(f"序列平稳化完成，使用 {self.args.method} 方法")
        
        if self.args.normalize:
            data = self.processor.normalize(method=self.args.norm_method)
            print(f"数据归一化完成，使用 {self.args.norm_method} 方法")
        
        print(f"数据预处理成功!")
    
    def predict(self):
        """进行预测"""
        if not self.processor:
            raise ValueError("请先加载数据")
        
        print(f"正在进行预测...")
        
        # 创建预测器
        self.predictor = TimeSeriesPredictor(
            self.processor.get_data(),
            self.processor.target_column
        )
        
        # 执行预测
        if self.args.algorithm == 'MA':
            forecasts = self.predictor.moving_average(
                window_size=self.args.window_size,
                forecast_steps=self.args.steps
            )
        elif self.args.algorithm == 'ES':
            forecasts = self.predictor.exponential_smoothing(
                alpha=self.args.alpha,
                forecast_steps=self.args.steps
            )
        elif self.args.algorithm == 'ARIMA':
            forecasts = self.predictor.arima(
                order=tuple(self.args.order),
                forecast_steps=self.args.steps
            )
        elif self.args.algorithm == 'Holt-Winters':
            forecasts = self.predictor.holt_winters(
                seasonal_periods=self.args.seasonal,
                forecast_steps=self.args.steps
            )
        elif self.args.algorithm == 'best':
            forecasts = self.predictor.predict_with_best_model(forecast_steps=self.args.steps)
        
        print(f"预测完成! 使用 {self.args.algorithm} 算法")
        print(f"未来 {self.args.steps} 步预测结果:")
        for i, val in enumerate(forecasts, 1):
            print(f"  第 {i} 步: {val:.4f}")
    
    def visualize(self):
        """可视化数据"""
        if not self.processor:
            raise ValueError("请先加载数据")
        
        print(f"正在生成可视化...")
        
        if self.args.type == 'ts':
            self.processor.visualize_time_series()
        elif self.args.type == 'trend':
            self.processor.visualize_trend()
        elif self.args.type == 'seasonality':
            self.processor.visualize_seasonality()
        elif self.args.type == 'histogram':
            self.processor.visualize_histogram()
        
        print(f"可视化完成!")
    
    def evaluate(self):
        """评估模型性能"""
        if not self.predictor:
            raise ValueError("请先进行预测")
        
        print(f"正在评估模型性能...")
        evaluations = self.predictor.evaluate()
        
        print("模型性能评估结果:")
        print("-" * 50)
        print(f"{'模型':<15}{'MSE':<15}{'RMSE':<15}{'MAE':<15}")
        print("-" * 50)
        for model, metrics in evaluations.items():
            print(f"{model:<15}{metrics['MSE']:<15.6f}{metrics['RMSE']:<15.6f}{metrics['MAE']:<15.6f}")
        
        best_model = self.predictor.get_best_model()
        print("-" * 50)
        print(f"最优模型: {best_model}")
    
    def show_info(self):
        """显示数据信息"""
        if not self.processor:
            raise ValueError("请先加载数据")
        
        self.processor.get_info()

if __name__ == "__main__":
    cli = SocratesCLI()
    cli.run()
