import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeSeriesPredictor:
    def __init__(self, data, target_column):
        self.data = data.copy()  # 深拷贝，避免SettingWithCopyWarning
        self.target_column = target_column
        self.models = {}
        self.predictions = {}
        self.evaluations = {}
        
        # 计算收益率
        self.data['return'] = self.data[self.target_column].pct_change().fillna(0)
        self.data['price_change'] = self.data[self.target_column].diff().fillna(0)
    
    def moving_average(self, window_size=3, forecast_steps=1):
        """移动平均预测法"""
        # 计算移动平均
        self.data['MA'] = self.data[self.target_column].rolling(window=window_size).mean()
        
        # 预测未来值
        last_ma = self.data['MA'].iloc[-1]
        forecasts = [last_ma] * forecast_steps
        
        self.models['MA'] = {'window_size': window_size}
        self.predictions['MA'] = forecasts
        
        return forecasts
    
    def predicted_moving_average(self, window_size=3, lookback_period=10, forecast_steps=1):
        """预测移动平均（PMA）模型"""
        # 计算移动平均
        self.data['PMA_ma'] = self.data[self.target_column].rolling(window=window_size).mean()
        
        # 计算线性回归斜率来预测趋势
        def calculate_slope(data, period):
            if len(data) < period:
                return 0
            x = np.arange(period)
            y = data[-period:]  # 当raw=True时，data已经是numpy数组，不需要.values
            # 使用线性回归计算斜率
            if np.std(x) == 0:  # 避免除以零
                return 0
            slope = np.cov(x, y)[0, 1] / np.var(x)
            return slope
        
        # 计算每个点的斜率
        self.data['PMA_slope'] = self.data[self.target_column].rolling(window=lookback_period).apply(
            lambda x: calculate_slope(x, lookback_period), raw=True
        )
        
        # PMA = 移动平均 + 斜率调整项
        self.data['PMA'] = self.data['PMA_ma'] + self.data['PMA_slope']
        
        # 预测未来值
        last_pma = self.data['PMA'].iloc[-1]
        last_slope = self.data['PMA_slope'].iloc[-1]
        
        # 未来预测值考虑斜率的持续影响
        forecasts = []
        current_value = last_pma
        for i in range(forecast_steps):
            forecasts.append(current_value)
            # 下一个预测值继续受斜率影响
            current_value += last_slope
        
        self.models['PMA'] = {'window_size': window_size, 'lookback_period': lookback_period}
        self.predictions['PMA'] = forecasts
        
        return forecasts
    
    def exponential_smoothing(self, alpha=0.2, forecast_steps=1):
        """指数平滑预测法"""
        # 初始化平滑值
        smoothed = [self.data[self.target_column].iloc[0]]
        
        # 计算指数平滑值
        for i in range(1, len(self.data)):
            val = alpha * self.data[self.target_column].iloc[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(val)
        
        self.data['ES'] = smoothed
        
        # 预测未来值
        last_es = smoothed[-1]
        forecasts = [last_es] * forecast_steps
        
        self.models['ES'] = {'alpha': alpha}
        self.predictions['ES'] = forecasts
        
        return forecasts
    
    def arima(self, order=(1, 1, 1), forecast_steps=1):
        """ARIMA预测法"""
        # 训练ARIMA模型
        model = ARIMA(self.data[self.target_column], order=order)
        model_fit = model.fit()
        
        # 预测未来值
        forecasts = model_fit.forecast(steps=forecast_steps)
        
        self.models['ARIMA'] = {'order': order, 'model': model_fit}
        self.predictions['ARIMA'] = forecasts.tolist()
        
        return forecasts.tolist()
    
    def holt_winters(self, seasonal_periods=12, trend='add', seasonal='add', forecast_steps=1):
        """Holt-Winters预测法（支持季节性）"""
        # 训练Holt-Winters模型
        model = ExponentialSmoothing(
            self.data[self.target_column],
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal
        )
        model_fit = model.fit()
        
        # 预测未来值
        forecasts = model_fit.forecast(steps=forecast_steps)
        
        self.models['Holt-Winters'] = {'model': model_fit}
        self.predictions['Holt-Winters'] = forecasts.tolist()
        
        return forecasts.tolist()
    
    def xgboost(self, feature_matrix=None, forecast_steps=1, predict_returns=True, n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, reg_alpha=0.0, reg_lambda=0.0, generate_interval=False, quantiles=[0.05, 0.5, 0.95]):
        """XGBoost预测模型，支持多特征输入和预测区间生成
        
        参数:
        feature_matrix: 特征矩阵，如果为None，则使用self.data作为特征
        forecast_steps: 预测步数
        predict_returns: 是否预测收益率，否则预测价格变化
        n_estimators: XGBoost估计器数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        random_state: 随机种子
        reg_alpha: L1正则化项
        reg_lambda: L2正则化项
        generate_interval: 是否生成预测区间
        quantiles: 用于生成预测区间的分位数列表
        """
        # 确定预测目标
        if predict_returns:
            target_name = 'return'
        else:
            target_name = 'price_change'
        
        if feature_matrix is None:
            # 如果没有提供特征矩阵，使用原始数据的滞后特征
            # 创建滞后特征
            lag_features = pd.DataFrame(index=self.data.index)
            lag_features[target_name] = self.data[target_name]
            lag_features['price'] = self.data[self.target_column]
            
            # 添加滞后特征（t-1, t-2, t-3）
            for i in range(1, 4):
                lag_features[f'lag_return_{i}'] = lag_features[target_name].shift(i)
                lag_features[f'lag_price_{i}'] = lag_features['price'].shift(i)
            
            lag_features.dropna(inplace=True)
            
            X = lag_features.drop(columns=[target_name, 'price'])
            y = lag_features[target_name]
        else:
            # 使用提供的特征矩阵
            # 添加收益率/价格变化目标
            feature_matrix_with_target = feature_matrix.copy()
            feature_matrix_with_target['return'] = feature_matrix[self.target_column].pct_change().fillna(0)
            feature_matrix_with_target['price_change'] = feature_matrix[self.target_column].diff().fillna(0)
            
            # 移除包含目标价格的列作为特征
            X = feature_matrix_with_target.drop(columns=[self.target_column, 'return', 'price_change'])
            y = feature_matrix_with_target[target_name]
        
        # 记录用于预测的特征
        model_features = list(X.columns)
        
        if generate_interval:
            # 生成预测区间
            quantile_models = {}
            forecasts_interval = {q: [] for q in quantiles}
            
            # 为每个分位数训练单独的模型
            for q in quantiles:
                model = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    objective='reg:quantileerror',
                    quantile_alpha=q
                )
                model.fit(X, y)
                quantile_models[q] = model
            
            # 预测未来值（带区间）
            current_X = X.iloc[-1:].copy()
            last_price = self.data[self.target_column].iloc[-1]
            
            # 存储每个时间步的预测区间
            interval_forecasts = []
            
            for step in range(forecast_steps):
                step_forecasts = {q: None for q in quantiles}
                
                # 为每个分位数进行预测
                for q in quantiles:
                    pred_change = quantile_models[q].predict(current_X)[0]
                    
                    # 转换为实际价格
                    if predict_returns:
                        pred_price = last_price * (1 + pred_change)
                    else:
                        pred_price = last_price + pred_change
                    
                    step_forecasts[q] = pred_price
                
                interval_forecasts.append(step_forecasts)
                
                # 更新滞后特征
                if feature_matrix is None:
                    # 更新滞后收益率
                    for i in range(2, 4):
                        current_X[f'lag_return_{i-1}'] = current_X[f'lag_return_{i}']
                    # 使用中位数预测作为下一次的变化
                    current_X[f'lag_return_3'] = quantile_models[0.5].predict(current_X)[0]
                    
                    # 更新滞后价格
                    for i in range(2, 4):
                        current_X[f'lag_price_{i-1}'] = current_X[f'lag_price_{i}']
                    current_X[f'lag_price_3'] = last_price
                else:
                    # 多特征矩阵情况，更新与目标价格相关的特征
                    # 使用中位数预测作为当前价格和变化
                    median_pred_change = quantile_models[0.5].predict(current_X)[0]
                    if predict_returns:
                        median_pred_price = last_price * (1 + median_pred_change)
                    else:
                        median_pred_price = last_price + median_pred_change
                    
                    for col in current_X.columns:
                        if 'lag_price' in col or 'price' in col.lower() and 'gold' not in col.lower() and 'usdcny' not in col.lower():
                            current_X[col] = last_price
                        elif 'return' in col.lower() or 'change' in col.lower():
                            current_X[col] = median_pred_change
                
                # 更新最后价格为中位数预测
                last_price = step_forecasts[0.5]
            
            # 提取中位数预测作为点预测
            forecasts = [f[0.5] for f in interval_forecasts]
            
            # 存储模型和预测结果
            self.models['XGBoost'] = {
                'model': quantile_models,
                'feature_matrix': feature_matrix,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'random_state': random_state,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'predict_returns': predict_returns,
                    'generate_interval': generate_interval,
                    'quantiles': quantiles
                },
                'model_features': model_features
            }
            
            self.predictions['XGBoost'] = {
                'point_forecast': forecasts,
                'interval_forecast': interval_forecasts
            }
            
            return forecasts, interval_forecasts
        else:
            # 标准点预测
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda
            )
            model.fit(X, y)
            
            # 预测未来值
            forecasts = []
            current_X = X.iloc[-1:].copy()
            last_price = self.data[self.target_column].iloc[-1]
            current_price = last_price
            
            for _ in range(forecast_steps):
                # 预测收益率或价格变化
                pred_change = model.predict(current_X)[0]
                
                # 转换为实际价格
                if predict_returns:
                    pred_price = current_price * (1 + pred_change)
                else:
                    pred_price = current_price + pred_change
                
                forecasts.append(pred_price)
                
                # 更新滞后特征
                if feature_matrix is None:
                    # 更新滞后收益率
                    for i in range(2, 4):
                        current_X[f'lag_return_{i-1}'] = current_X[f'lag_return_{i}']
                    current_X[f'lag_return_3'] = pred_change
                    
                    # 更新滞后价格
                    for i in range(2, 4):
                        current_X[f'lag_price_{i-1}'] = current_X[f'lag_price_{i}']
                    current_X[f'lag_price_3'] = current_price
                else:
                    # 多特征矩阵情况，更新与目标价格相关的特征
                    for col in current_X.columns:
                        if 'lag_price' in col or 'price' in col.lower() and 'gold' not in col.lower() and 'usdcny' not in col.lower():
                            current_X[col] = current_price
                        elif 'return' in col.lower() or 'change' in col.lower():
                            current_X[col] = pred_change
                
                # 更新当前价格用于下一次预测
                current_price = pred_price
            
            self.models['XGBoost'] = {
                'model': model,
                'feature_matrix': feature_matrix,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'random_state': random_state,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'predict_returns': predict_returns,
                    'generate_interval': generate_interval
                },
                'model_features': model_features
            }
            self.predictions['XGBoost'] = forecasts
            
            return forecasts
    
    def lstm(self, lookback=10, forecast_steps=1, units=16, epochs=30, batch_size=32, dropout=0.3, feature_matrix=None):
        """LSTM神经网络预测模型
        
        参数:
        lookback: 用于预测的历史序列长度
        forecast_steps: 预测步数
        units: LSTM层的神经元数量
        epochs: 训练轮数
        batch_size: 批次大小
        dropout: Dropout率
        feature_matrix: 特征矩阵，如果为None，则使用self.data的目标列
        """
        # 确定预测目标
        if hasattr(self, 'data') and 'return' in self.data.columns and 'price_change' in self.data.columns:
            # 如果已经计算了收益率和价格变化，使用收益率作为目标
            target = self.data['return'].values.reshape(-1, 1)
        else:
            # 否则使用原始价格
            target = self.data[self.target_column].values.reshape(-1, 1)
        
        if feature_matrix is None:
            # 如果没有提供特征矩阵，使用目标列的历史数据作为特征
            features = target
            is_multivariate = False
        else:
            # 使用提供的特征矩阵
            features = feature_matrix.values
            is_multivariate = True
        
        # 合并特征和目标
        if is_multivariate:
            # 多变量情况下，将目标添加到特征矩阵的第一列
            data = np.hstack((target, features))
        else:
            data = target
        
        # 分割训练集和测试集（使用最后20%作为测试集）
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 只使用训练集数据来拟合缩放器（避免数据泄露）
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        
        # 转换训练集和测试集
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        
        # 创建训练数据序列
        def create_sequences(data, lookback):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i, 0])  # 始终预测第一列（目标列）
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(scaled_train_data, lookback)
        
        # 构建更简单的LSTM模型
        model = Sequential()
        model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        
        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 训练模型，使用验证集监控性能
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0, 
            validation_split=0.2
        )
        
        # 预测未来值
        forecasts = []
        # 转换完整数据集（用于生成预测序列）
        scaled_full_data = scaler.transform(data)
        # 使用完整数据集的最后lookback个数据点作为初始序列
        current_sequence = scaled_full_data[-lookback:].reshape(1, lookback, -1)
        
        for _ in range(forecast_steps):
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # 更新序列
            if is_multivariate:
                # 对于多变量情况，只更新目标列，其他列保持不变
                next_pred_full = np.zeros(scaled_full_data.shape[1])
                next_pred_full[0] = next_pred  # 目标列是第一列
                next_pred_full[1:] = current_sequence[0, -1, 1:]  # 其他列使用上一个序列的最后一行
                next_sequence = np.append(current_sequence[:, 1:, :], [[next_pred_full]], axis=1)
            else:
                # 对于单变量情况，直接添加预测值
                next_pred = next_pred.reshape(1, 1, 1)
                next_sequence = np.append(current_sequence[:, 1:, :], next_pred, axis=1)
            
            current_sequence = next_sequence
        
        # 逆缩放预测值
        if is_multivariate:
            # 多变量情况，需要构造完整的特征向量进行逆缩放
            scaled_forecasts = np.zeros((forecast_steps, scaled_full_data.shape[1]))
            scaled_forecasts[:, 0] = forecasts
            # 其他特征使用最后已知值
            scaled_forecasts[:, 1:] = scaled_full_data[-1, 1:]
            forecasts_inv = scaler.inverse_transform(scaled_forecasts)[:, 0]
        else:
            # 单变量情况
            forecasts_inv = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        self.models['LSTM'] = {
            'model': model,
            'scaler': scaler,
            'lookback': lookback,
            'is_multivariate': is_multivariate,
            'feature_matrix': feature_matrix,
            'params': {
                'units': units,
                'epochs': epochs,
                'batch_size': batch_size,
                'dropout': dropout
            }
        }
        self.predictions['LSTM'] = forecasts_inv.tolist()
        
        return forecasts_inv.tolist()
    
    def evaluate(self, test_data=None, feature_matrix=None):
        """评估模型性能
        
        参数:
        test_data: 测试集数据，如果为None，则使用最后20%作为测试集
        feature_matrix: 特征矩阵，用于XGBoost等基于特征的模型
        """
        if test_data is None:
            # 使用最后几个值作为测试集
            test_size = int(len(self.data) * 0.2)
            train_data = self.data[:-test_size]
            test_data = self.data[-test_size:]
        else:
            # 如果提供了测试集，使用整个数据作为训练集
            train_data = self.data
        
        for model_name in self.models.keys():
            if model_name == 'MA':
                window_size = self.models[model_name]['window_size']
                # 使用相同参数重新预测测试集
                train_ma = train_data[self.target_column].rolling(window=window_size).mean()
                last_ma = train_ma.iloc[-1]
                predictions = [last_ma] * len(test_data)
            elif model_name == 'PMA':
                window_size = self.models[model_name]['window_size']
                lookback_period = self.models[model_name]['lookback_period']
                
                # 重新计算PMA相关值
                train_ma = train_data[self.target_column].rolling(window=window_size).mean()
                
                def calculate_slope(data, period):
                    if len(data) < period:
                        return 0
                    x = np.arange(period)
                    y = data[-period:]  # data是Series，需要.values转为numpy数组
                    if np.std(x) == 0:
                        return 0
                    slope = np.cov(x, y)[0, 1] / np.var(x)
                    return slope
                
                # 计算训练集的斜率
                train_slope = train_data[self.target_column].rolling(window=lookback_period).apply(
                    lambda x: calculate_slope(x, lookback_period), raw=True
                )
                
                train_pma = train_ma + train_slope
                last_pma = train_pma.iloc[-1]
                last_slope = train_slope.iloc[-1]
                
                # 生成预测
                predictions = []
                current_value = last_pma
                for i in range(len(test_data)):
                    predictions.append(current_value)
                    current_value += last_slope
            elif model_name == 'ES':
                alpha = self.models[model_name]['alpha']
                # 使用相同参数重新预测测试集
                smoothed = [train_data[self.target_column].iloc[0]]
                for i in range(1, len(train_data)):
                    val = alpha * train_data[self.target_column].iloc[i] + (1 - alpha) * smoothed[-1]
                    smoothed.append(val)
                predictions = [smoothed[-1]] * len(test_data)
            elif model_name == 'ARIMA':
                # 使用训练好的模型预测测试集
                model_fit = self.models[model_name]['model']
                predictions = model_fit.forecast(steps=len(test_data)).tolist()
            elif model_name == 'Holt-Winters':
                # 使用训练好的模型预测测试集
                model_fit = self.models[model_name]['model']
                predictions = model_fit.forecast(steps=len(test_data)).tolist()
            elif model_name == 'XGBoost':
                # 处理XGBoost模型的评估
                model_params = self.models[model_name]['params']
                model_feature_matrix = self.models[model_name]['feature_matrix']
                predict_returns = model_params.get('predict_returns', True)
                
                # 确定预测目标
                if predict_returns:
                    target_name = 'return'
                else:
                    target_name = 'price_change'
                
                if model_feature_matrix is None:
                    # 如果没有提供特征矩阵，使用滞后特征
                    # 创建训练集和测试集的滞后特征
                    def create_lag_features(data):
                        lag_features = pd.DataFrame(index=data.index)
                        lag_features[target_name] = data[target_name]
                        lag_features['price'] = data[self.target_column]
                        
                        for i in range(1, 4):
                            lag_features[f'lag_return_{i}'] = lag_features[target_name].shift(i)
                            lag_features[f'lag_price_{i}'] = lag_features['price'].shift(i)
                        
                        lag_features.dropna(inplace=True)
                        return lag_features
                    
                    # 确保训练集和测试集有收益率/价格变化列
                    train_data_eval = train_data.copy()
                    test_data_eval = test_data.copy()
                    
                    train_data_eval['return'] = train_data_eval[self.target_column].pct_change().fillna(0)
                    train_data_eval['price_change'] = train_data_eval[self.target_column].diff().fillna(0)
                    
                    test_data_eval['return'] = test_data_eval[self.target_column].pct_change().fillna(0)
                    test_data_eval['price_change'] = test_data_eval[self.target_column].diff().fillna(0)
                    
                    train_lag = create_lag_features(train_data_eval)
                    test_lag = create_lag_features(test_data_eval)
                    
                    # 确保测试集有足够的滞后特征
                    if len(test_lag) > 0:
                        X_train = train_lag.drop(columns=[target_name, 'price'])
                        y_train = train_lag[target_name]
                        X_test = test_lag.drop(columns=[target_name, 'price'])
                        y_test = test_lag[target_name]
                        
                        # 重新训练模型
                        model = XGBRegressor(**model_params)
                        model.fit(X_train, y_train)
                        
                        # 预测
                        predictions = model.predict(X_test).tolist()
                        
                        # 将收益率/价格变化转换为价格预测
                        price_predictions = []
                        last_train_price = train_data_eval[self.target_column].iloc[-1]
                        current_price = last_train_price
                        
                        for pred_change in predictions:
                            if predict_returns:
                                next_price = current_price * (1 + pred_change)
                            else:
                                next_price = current_price + pred_change
                            price_predictions.append(next_price)
                            current_price = next_price
                        
                        predictions = price_predictions
                        
                        # 调整预测长度以匹配测试集
                        if len(predictions) < len(test_data):
                            last_pred = predictions[-1] if predictions else train_data[self.target_column].iloc[-1]
                            predictions += [last_pred] * (len(test_data) - len(predictions))
                        elif len(predictions) > len(test_data):
                            predictions = predictions[:len(test_data)]
                    else:
                        # 如果测试集没有足够的滞后特征，使用简单预测
                        predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
                else:
                    # 使用提供的特征矩阵
                    if feature_matrix is None:
                        feature_matrix = model_feature_matrix
                    
                    # 分割特征矩阵为训练集和测试集
                    train_index = train_data.index
                    test_index = test_data.index
                    
                    # 确保索引在特征矩阵中
                    common_train_index = train_index.intersection(feature_matrix.index)
                    common_test_index = test_index.intersection(feature_matrix.index)
                    
                    if len(common_test_index) > 0:
                        # 准备训练集和测试集特征矩阵
                        train_feature_matrix = feature_matrix.loc[common_train_index].copy()
                        test_feature_matrix = feature_matrix.loc[common_test_index].copy()
                        
                        # 添加收益率/价格变化目标
                        train_feature_matrix['return'] = train_feature_matrix[self.target_column].pct_change().fillna(0)
                        train_feature_matrix['price_change'] = train_feature_matrix[self.target_column].diff().fillna(0)
                        
                        test_feature_matrix['return'] = test_feature_matrix[self.target_column].pct_change().fillna(0)
                        test_feature_matrix['price_change'] = test_feature_matrix[self.target_column].diff().fillna(0)
                        
                        # 准备特征和目标
                        X_train = train_feature_matrix.drop(columns=[self.target_column, 'return', 'price_change'])
                        y_train = train_feature_matrix[target_name]
                        X_test = test_feature_matrix.drop(columns=[self.target_column, 'return', 'price_change'])
                        y_test = test_feature_matrix[target_name]
                        
                        # 重新训练模型
                        model = XGBRegressor(**model_params)
                        model.fit(X_train, y_train)
                        
                        # 预测
                        predictions = model.predict(X_test).tolist()
                        
                        # 将收益率/价格变化转换为价格预测
                        price_predictions = []
                        last_train_price = train_feature_matrix[self.target_column].iloc[-1]
                        current_price = last_train_price
                        
                        for pred_change in predictions:
                            if predict_returns:
                                next_price = current_price * (1 + pred_change)
                            else:
                                next_price = current_price + pred_change
                            price_predictions.append(next_price)
                            current_price = next_price
                        
                        predictions = price_predictions
                        
                        # 调整预测长度以匹配测试集
                        if len(predictions) < len(test_data):
                            last_pred = predictions[-1] if predictions else train_data[self.target_column].iloc[-1]
                            predictions += [last_pred] * (len(test_data) - len(predictions))
                        elif len(predictions) > len(test_data):
                            predictions = predictions[:len(test_data)]
                    else:
                        # 如果测试集没有对应的特征，使用简单预测
                        predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
            elif model_name == 'LSTM':
                # 处理LSTM模型的评估
                model_params = self.models[model_name]['params']
                lookback = self.models[model_name]['lookback']
                is_multivariate = self.models[model_name]['is_multivariate']
                model_feature_matrix = self.models[model_name]['feature_matrix']
                
                # 准备数据
                if is_multivariate:
                    if feature_matrix is None:
                        feature_matrix = model_feature_matrix
                    
                    if feature_matrix is None:
                        # 如果没有特征矩阵，无法进行LSTM评估，使用简单预测
                        predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
                    else:
                        # 分割特征矩阵为训练集和测试集
                        train_index = train_data.index
                        test_index = test_data.index
                        
                        # 确保索引在特征矩阵中
                        common_train_index = train_index.intersection(feature_matrix.index)
                        common_test_index = test_index.intersection(feature_matrix.index)
                        
                        if len(common_train_index) > lookback and len(common_test_index) > 0:
                            # 准备训练数据
                            train_data_full = feature_matrix.loc[common_train_index]
                            X_train = []
                            y_train = []
                            
                            # 创建序列
                            def create_sequences(data, lookback):
                                X, y = [], []
                                for i in range(lookback, len(data)):
                                    X.append(data[i-lookback:i])  # data已经是numpy数组，不需要.values
                                    y.append(data[i, 0])
                                return np.array(X), np.array(y)
                            
                            # 数据缩放
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            scaled_train = scaler.fit_transform(train_data_full.values)
                            X_train, y_train = create_sequences(scaled_train, lookback)
                            
                            # 构建LSTM模型
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import LSTM, Dense, Dropout
                            
                            model = Sequential()
                            model.add(LSTM(units=model_params['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                            model.add(Dropout(model_params['dropout']))
                            model.add(LSTM(units=model_params['units'], return_sequences=False))
                            model.add(Dropout(model_params['dropout']))
                            model.add(Dense(units=25))
                            model.add(Dense(units=1))
                            
                            # 编译模型
                            model.compile(optimizer='adam', loss='mean_squared_error')
                            
                            # 训练模型
                            model.fit(X_train, y_train, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
                            
                            # 准备测试数据
                            full_data = pd.concat([train_data_full, feature_matrix.loc[common_test_index]])
                            scaled_full = scaler.transform(full_data.values)
                            
                            # 创建测试序列
                            test_sequences = []
                            for i in range(len(scaled_train) - lookback + 1, len(scaled_full) - lookback + 1):
                                test_sequences.append(scaled_full[i:i+lookback])
                            
                            if test_sequences:
                                test_sequences = np.array(test_sequences)
                                
                                # 预测
                                scaled_predictions = model.predict(test_sequences, verbose=0)
                                
                                # 逆缩放
                                scaled_predictions_full = np.zeros((len(scaled_predictions), scaled_full.shape[1]))
                                scaled_predictions_full[:, 0] = scaled_predictions.flatten()
                                predictions = scaler.inverse_transform(scaled_predictions_full)[:, 0]
                                
                                # 调整预测长度以匹配测试集
                                if len(predictions) < len(test_data):
                                    last_pred = predictions[-1] if len(predictions) > 0 else np.mean(y_train)
                                    predictions = np.concatenate([predictions, [last_pred] * (len(test_data) - len(predictions))])
                                elif len(predictions) > len(test_data):
                                    predictions = predictions[:len(test_data)]
                                
                                predictions = predictions.tolist()
                            else:
                                predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
                        else:
                            predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
                else:
                    # 单变量LSTM
                    # 准备训练数据
                    train_close = train_data[self.target_column].values.reshape(-1, 1)
                    
                    if len(train_close) > lookback:
                        # 数据缩放
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_train = scaler.fit_transform(train_close)
                        
                        # 创建序列
                        X_train = []
                        y_train = []
                        for i in range(lookback, len(scaled_train)):
                            X_train.append(scaled_train[i-lookback:i, 0])
                            y_train.append(scaled_train[i, 0])
                        
                        X_train, y_train = np.array(X_train), np.array(y_train)
                        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                        
                        # 构建LSTM模型
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import LSTM, Dense, Dropout
                        
                        model = Sequential()
                        model.add(LSTM(units=model_params['units'], return_sequences=True, input_shape=(X_train.shape[1], 1)))
                        model.add(Dropout(model_params['dropout']))
                        model.add(LSTM(units=model_params['units'], return_sequences=False))
                        model.add(Dropout(model_params['dropout']))
                        model.add(Dense(units=25))
                        model.add(Dense(units=1))
                        
                        # 编译模型
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # 训练模型
                        model.fit(X_train, y_train, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
                        
                        # 准备测试数据
                        test_close = test_data[self.target_column].values.reshape(-1, 1)
                        full_data = np.concatenate([train_close, test_close])
                        scaled_full = scaler.transform(full_data)
                        
                        # 创建测试序列
                        test_sequences = []
                        for i in range(len(scaled_train) - lookback + 1, len(scaled_full) - lookback + 1):
                            test_sequences.append(scaled_full[i:i+lookback, 0])
                        
                        if test_sequences:
                            test_sequences = np.array(test_sequences)
                            test_sequences = np.reshape(test_sequences, (test_sequences.shape[0], test_sequences.shape[1], 1))
                            
                            # 预测
                            scaled_predictions = model.predict(test_sequences, verbose=0)
                            predictions = scaler.inverse_transform(scaled_predictions).flatten()
                            
                            # 调整预测长度以匹配测试集
                            if len(predictions) < len(test_data):
                                last_pred = predictions[-1] if len(predictions) > 0 else np.mean(y_train)
                                predictions = np.concatenate([predictions, [last_pred] * (len(test_data) - len(predictions))])
                            elif len(predictions) > len(test_data):
                                predictions = predictions[:len(test_data)]
                            
                            predictions = predictions.tolist()
                        else:
                            predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
                    else:
                        predictions = [train_data[self.target_column].iloc[-1]] * len(test_data)
            
            # 计算评估指标
            mse = mean_squared_error(test_data[self.target_column], predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data[self.target_column], predictions)
            
            self.evaluations[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }
        
        return self.evaluations
    
    def get_best_model(self):
        """选择最优模型"""
        if not self.evaluations:
            self.evaluate()
        
        # 选择RMSE最小的模型
        best_model = min(self.evaluations, key=lambda x: self.evaluations[x]['RMSE'])
        return best_model
    
    def predict_with_best_model(self, forecast_steps=1, feature_matrix=None):
        """使用最优模型进行预测
        
        参数:
        forecast_steps: 预测步数
        feature_matrix: 特征矩阵，用于XGBoost等基于特征的模型
        """
        best_model = self.get_best_model()
        
        if best_model == 'MA':
            return self.moving_average(self.models[best_model]['window_size'], forecast_steps)
        elif best_model == 'PMA':
            return self.predicted_moving_average(
                self.models[best_model]['window_size'], 
                self.models[best_model]['lookback_period'], 
                forecast_steps
            )
        elif best_model == 'ES':
            return self.exponential_smoothing(self.models[best_model]['alpha'], forecast_steps)
        elif best_model == 'ARIMA':
            return self.arima(self.models[best_model]['order'], forecast_steps)
        elif best_model == 'Holt-Winters':
            return self.holt_winters(seasonal_periods=12, forecast_steps=forecast_steps)
        elif best_model == 'XGBoost':
            params = self.models[best_model]['params']
            return self.xgboost(
                feature_matrix=feature_matrix,
                forecast_steps=forecast_steps,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=params['random_state']
            )
    
    def grid_search(self, model_type, param_grid, forecast_steps=1, cv=5, feature_matrix=None):
        """网格搜索寻找最优参数
        
        Args:
            model_type: 模型类型 ('MA', 'PMA', 'ES', 'XGBoost')
            param_grid: 参数网格，如{'window_size': [3, 5, 7], 'lookback_period': [5, 10]}
            forecast_steps: 预测步数
            cv: 交叉验证折数
            feature_matrix: 特征矩阵，用于XGBoost等基于特征的模型
        
        Returns:
            最优参数和对应的最小误差
        """
        import itertools
        import numpy as np
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        best_params = None
        best_score = float('inf')
        
        # 分割数据用于交叉验证
        def time_series_cv_split(data, n_splits):
            folds = []
            split_size = len(data) // n_splits
            for i in range(n_splits):
                test_start = (i + 1) * split_size
                train = data[:test_start]
                test = data[test_start:test_start + forecast_steps]
                if len(test) == forecast_steps:
                    folds.append((train, test))
            return folds
        
        folds = time_series_cv_split(self.data, cv)
        
        # 遍历所有参数组合
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            fold_scores = []
            
            # 进行交叉验证
            for train, test in folds:
                if model_type == 'XGBoost':
                    # XGBoost需要特殊处理
                    if feature_matrix is None:
                        # 创建滞后特征
                        def create_lag_features(data):
                            lag_features = pd.DataFrame(index=data.index)
                            lag_features['target'] = data[self.target_column]
                            for i in range(1, 4):
                                lag_features[f'lag_{i}'] = lag_features['target'].shift(i)
                            lag_features.dropna(inplace=True)
                            return lag_features
                        
                        train_lag = create_lag_features(train)
                        test_lag = create_lag_features(test)
                        
                        if len(train_lag) > 0 and len(test_lag) > 0:
                            X_train = train_lag.drop(columns=['target'])
                            y_train = train_lag['target']
                            X_test = test_lag.drop(columns=['target'])
                            y_test = test_lag['target']
                            
                            # 训练XGBoost模型
                            model = XGBRegressor(**param_dict)
                            model.fit(X_train, y_train)
                            
                            # 预测
                            predictions = model.predict(X_test)
                            actual = y_test.values
                            
                            if len(predictions) == len(actual):
                                mse = mean_squared_error(actual, predictions)
                                fold_scores.append(mse)
                    else:
                        # 使用提供的特征矩阵
                        train_index = train.index
                        test_index = test.index
                        
                        # 确保索引在特征矩阵中
                        common_train_index = train_index.intersection(feature_matrix.index)
                        common_test_index = test_index.intersection(feature_matrix.index)
                        
                        if len(common_train_index) > 0 and len(common_test_index) > 0:
                            X_train = feature_matrix.loc[common_train_index].drop(columns=[self.target_column])
                            y_train = feature_matrix.loc[common_train_index][self.target_column]
                            X_test = feature_matrix.loc[common_test_index].drop(columns=[self.target_column])
                            y_test = feature_matrix.loc[common_test_index][self.target_column]
                            
                            # 训练XGBoost模型
                            model = XGBRegressor(**param_dict)
                            model.fit(X_train, y_train)
                            
                            # 预测
                            predictions = model.predict(X_test)
                            actual = y_test.values
                            
                            if len(predictions) == len(actual):
                                mse = mean_squared_error(actual, predictions)
                                fold_scores.append(mse)
                else:
                    # 创建临时预测器
                    temp_predictor = TimeSeriesPredictor(train, self.target_column)
                    
                    # 训练模型
                    if model_type == 'MA':
                        temp_predictor.moving_average(
                            window_size=param_dict['window_size'], 
                            forecast_steps=forecast_steps
                        )
                    elif model_type == 'PMA':
                        temp_predictor.predicted_moving_average(
                            window_size=param_dict['window_size'], 
                            lookback_period=param_dict['lookback_period'], 
                            forecast_steps=forecast_steps
                        )
                    elif model_type == 'ES':
                        temp_predictor.exponential_smoothing(
                            alpha=param_dict['alpha'], 
                            forecast_steps=forecast_steps
                        )
                    
                    # 计算预测误差
                    predictions = temp_predictor.predictions[model_type]
                    actual = test[self.target_column].values
                    
                    if len(predictions) == len(actual):
                        mse = mean_squared_error(actual, predictions)
                        fold_scores.append(mse)
            
            # 计算平均误差
            if fold_scores:
                avg_score = np.mean(fold_scores)
                
                # 更新最优参数
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = param_dict
        
        return best_params, best_score
