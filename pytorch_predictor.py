import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
import logging

class PyTorchTimeSeriesPredictor:
    def __init__(self, data, target_column):
        self.data = data.copy()
        self.target_column = target_column
        self.models = {}
        self.predictions = {}
        self.evaluations = {}
        self.scalers = {}
        
        # 确保使用GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {self.device}")
        
        # 计算收益率和价格变化
        self.data['return'] = self.data[self.target_column].pct_change().fillna(0)
        self.data['price_change'] = self.data[self.target_column].diff().fillna(0)
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM层
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
            
            # 注意力层
            self.attention = nn.Linear(hidden_size, 1)
            
            # 全连接层
            self.fc = nn.Linear(hidden_size, output_size)
            
            # Dropout层
            self.dropout = nn.Dropout(dropout_prob)
        
        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # 前向传播 LSTM
            out, _ = self.lstm(x, (h0, c0))
            
            # 应用注意力机制
            attn_weights = torch.softmax(self.attention(out), dim=1)
            context = torch.sum(attn_weights * out, dim=1)
            
            # 应用dropout和全连接层
            out = self.dropout(context)
            out = self.fc(out)
            
            return out
    
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # GRU层
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
            
            # 全连接层
            self.fc = nn.Linear(hidden_size, output_size)
            
            # Dropout层
            self.dropout = nn.Dropout(dropout_prob)
        
        def forward(self, x):
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # 前向传播 GRU
            out, _ = self.gru(x, h0)
            
            # 取最后一个时间步的输出
            out = out[:, -1, :]
            
            # 应用dropout和全连接层
            out = self.dropout(out)
            out = self.fc(out)
            
            return out
    
    class TransformerModel(nn.Module):
        def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout_prob=0.1):
            super().__init__()
            
            # 输入嵌入层
            self.embedding = nn.Linear(input_size, d_model)
            
            # Transformer层
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout_prob
            )
            
            # 输出层
            self.fc = nn.Linear(d_model, output_size)
            
            # Dropout层
            self.dropout = nn.Dropout(dropout_prob)
        
        def forward(self, src, tgt):
            # 嵌入层
            src = self.embedding(src)
            tgt = self.embedding(tgt)
            
            # 添加位置编码
            src = src + self.positional_encoding(src)
            tgt = tgt + self.positional_encoding(tgt)
            
            # 应用dropout
            src = self.dropout(src)
            tgt = self.dropout(tgt)
            
            # 转换为Transformer所需的形状 (seq_len, batch, feature)
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
            
            # 生成掩码
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
            
            # 前向传播
            out = self.transformer(src, tgt, tgt_mask=tgt_mask)
            
            # 转换回原始形状并应用输出层
            out = out.permute(1, 0, 2)
            out = self.fc(out[:, -1, :])
            
            return out
        
        def positional_encoding(self, x):
            seq_len, d_model = x.size(1), x.size(2)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pos_encoding = torch.zeros(x.size(0), seq_len, d_model)
            pos_encoding[:, :, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, :, 1::2] = torch.cos(position * div_term)
            return pos_encoding.to(x.device)
    
    class TFTModel(nn.Module):
        """Temporal Fusion Transformer (TFT)模型，专为时间序列预测设计"""
        def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=2, dropout_prob=0.1, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            
            # 输入嵌入层
            self.input_embedding = nn.Linear(input_size, hidden_size)
            
            # 门控残差网络 (GRN) 用于变量选择
            self.grn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )
            
            # 多头注意力层
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob)
            
            # 前馈网络
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            
            # 层归一化
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)
            
            # 门控循环单元
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
            
            # 输出层
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, output_size)
            )
            
            # Dropout层
            self.dropout = nn.Dropout(dropout_prob)
        
        def forward(self, x):
            # 输入嵌入
            embedded = self.input_embedding(x)
            
            # 变量选择门控
            variable_weights = self.grn(embedded)
            gated_input = embedded * variable_weights
            
            # 多头注意力
            attn_output, _ = self.attention(gated_input, gated_input, gated_input)
            attn_output = self.dropout(attn_output)
            attn_output = self.layer_norm1(gated_input + attn_output)
            
            # 前馈网络
            ff_output = self.feed_forward(attn_output)
            ff_output = self.dropout(ff_output)
            transformer_output = self.layer_norm2(attn_output + ff_output)
            
            # GRU层
            gru_output, _ = self.gru(transformer_output)
            
            # 取最后一个时间步的输出
            last_output = gru_output[:, -1, :]
            
            # 输出层
            out = self.output_layer(last_output)
            
            return out
    
    def create_sequences(self, data, lookback, forecast_horizon):
        """创建时间序列预测的输入输出序列"""
        X, y = [], []
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:(i+lookback)])
            y.append(data[(i+lookback):(i+lookback+forecast_horizon), 0])
        return np.array(X), np.array(y)
    
    def train_model(self, model, train_loader, test_loader, criterion, optimizer, epochs=100, patience=10):
        """训练模型，包含早停机制"""
        model.to(self.device)
        
        # 记录最佳验证损失
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # 前向传播
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(test_loader.dataset)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"早停于第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model.pth'))
        model.to(self.device)
        
        return model
    
    def lstm_pytorch(self, lookback=20, forecast_steps=1, hidden_size=64, num_layers=2, dropout_prob=0.3, epochs=100, batch_size=32, learning_rate=0.001, feature_matrix=None):
        """PyTorch LSTM预测模型"""
        logging.info("开始训练PyTorch LSTM模型...")
        
        # 确定输入数据
        if feature_matrix is None:
            # 使用目标列的收益率作为输入
            data = self.data['return'].values.reshape(-1, 1)
            input_size = 1
        else:
            # 使用提供的特征矩阵
            data = feature_matrix.values
            input_size = data.shape[1]
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        self.scalers['lstm_pytorch'] = scaler
        
        # 创建训练和测试序列
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        X_train, y_train = self.create_sequences(train_data, lookback, 1)
        X_test, y_test = self.create_sequences(test_data, lookback, 1)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = self.LSTMModel(input_size, hidden_size, num_layers, 1, dropout_prob)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        model = self.train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience=10)
        
        # 预测未来值
        forecasts = []
        
        # 使用最后一个lookback长度的序列作为初始输入
        current_sequence = data_scaled[-lookback:].reshape(1, lookback, input_size)
        current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        for _ in range(forecast_steps):
            # 预测
            with torch.no_grad():
                model.eval()
                next_pred = model(current_sequence_tensor)
                next_pred_np = next_pred.cpu().numpy()[0, 0]
            
            forecasts.append(next_pred_np)
            
            # 更新序列
            next_pred_reshaped = np.array([[next_pred_np]]) if input_size == 1 else np.repeat([[next_pred_np]], input_size, axis=1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped.reshape(1, 1, input_size), axis=1)
            current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        # 逆归一化预测结果
        if input_size == 1:
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            forecasts_inv = scaler.inverse_transform(forecasts_reshaped).flatten()
        else:
            # 多变量情况，只逆归一化第一列（目标列）
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            # 创建全零数组，只填第一列
            full_forecasts = np.zeros((len(forecasts), input_size))
            full_forecasts[:, 0] = forecasts_reshaped.flatten()
            forecasts_inv = scaler.inverse_transform(full_forecasts)[:, 0]
        
        # 转换为实际价格预测
        last_price = self.data[self.target_column].iloc[-1]
        price_forecasts = []
        current_price = last_price
        
        for pred_return in forecasts_inv:
            pred_price = current_price * (1 + pred_return)
            price_forecasts.append(pred_price)
            current_price = pred_price
        
        self.models['lstm_pytorch'] = {
            'model': model,
            'params': {
                'lookback': lookback,
                'forecast_steps': forecast_steps,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout_prob': dropout_prob,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        self.predictions['lstm_pytorch'] = price_forecasts
        
        logging.info("PyTorch LSTM模型训练完成")
        
        return price_forecasts
    
    def gru_pytorch(self, lookback=20, forecast_steps=1, hidden_size=64, num_layers=2, dropout_prob=0.3, epochs=100, batch_size=32, learning_rate=0.001, feature_matrix=None):
        """PyTorch GRU预测模型"""
        logging.info("开始训练PyTorch GRU模型...")
        
        # 确定输入数据
        if feature_matrix is None:
            # 使用目标列的收益率作为输入
            data = self.data['return'].values.reshape(-1, 1)
            input_size = 1
        else:
            # 使用提供的特征矩阵
            data = feature_matrix.values
            input_size = data.shape[1]
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        self.scalers['gru_pytorch'] = scaler
        
        # 创建训练和测试序列
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        X_train, y_train = self.create_sequences(train_data, lookback, 1)
        X_test, y_test = self.create_sequences(test_data, lookback, 1)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = self.GRUModel(input_size, hidden_size, num_layers, 1, dropout_prob)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        model = self.train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience=10)
        
        # 预测未来值
        forecasts = []
        
        # 使用最后一个lookback长度的序列作为初始输入
        current_sequence = data_scaled[-lookback:].reshape(1, lookback, input_size)
        current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        for _ in range(forecast_steps):
            # 预测
            with torch.no_grad():
                model.eval()
                next_pred = model(current_sequence_tensor)
                next_pred_np = next_pred.cpu().numpy()[0, 0]
            
            forecasts.append(next_pred_np)
            
            # 更新序列
            next_pred_reshaped = np.array([[next_pred_np]]) if input_size == 1 else np.repeat([[next_pred_np]], input_size, axis=1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped.reshape(1, 1, input_size), axis=1)
            current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        # 逆归一化预测结果
        if input_size == 1:
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            forecasts_inv = scaler.inverse_transform(forecasts_reshaped).flatten()
        else:
            # 多变量情况，只逆归一化第一列（目标列）
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            # 创建全零数组，只填第一列
            full_forecasts = np.zeros((len(forecasts), input_size))
            full_forecasts[:, 0] = forecasts_reshaped.flatten()
            forecasts_inv = scaler.inverse_transform(full_forecasts)[:, 0]
        
        # 转换为实际价格预测
        last_price = self.data[self.target_column].iloc[-1]
        price_forecasts = []
        current_price = last_price
        
        for pred_return in forecasts_inv:
            pred_price = current_price * (1 + pred_return)
            price_forecasts.append(pred_price)
            current_price = pred_price
        
        self.models['gru_pytorch'] = {
            'model': model,
            'params': {
                'lookback': lookback,
                'forecast_steps': forecast_steps,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout_prob': dropout_prob,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        self.predictions['gru_pytorch'] = price_forecasts
        
        logging.info("PyTorch GRU模型训练完成")
        
        return price_forecasts
    
    def transformer_pytorch(self, lookback=20, forecast_steps=1, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout_prob=0.1, epochs=100, batch_size=32, learning_rate=0.001, feature_matrix=None):
        """PyTorch Transformer预测模型"""
        logging.info("开始训练PyTorch Transformer模型...")
        
        # 确定输入数据
        if feature_matrix is None:
            # 使用目标列的收益率作为输入
            data = self.data['return'].values.reshape(-1, 1)
            input_size = 1
        else:
            # 使用提供的特征矩阵
            data = feature_matrix.values
            input_size = data.shape[1]
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        self.scalers['transformer_pytorch'] = scaler
        
        # 创建训练和测试数据
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        # 创建Transformer所需的序列
        def create_transformer_sequences(data, lookback):
            encoder_inputs, decoder_inputs, y = [], [], []
            for i in range(len(data) - lookback - 1):
                # 编码器输入：历史序列
                encoder_input = data[i:(i+lookback)]
                # 解码器输入：目标序列（包含预测点的前一个点）
                decoder_input = data[(i+lookback-1):(i+lookback)]
                # 目标输出：预测点
                target = data[(i+lookback), 0]
                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                y.append(target)
            return np.array(encoder_inputs), np.array(decoder_inputs), np.array(y)
        
        X_train_encoder, X_train_decoder, y_train = create_transformer_sequences(train_data, lookback)
        X_test_encoder, X_test_decoder, y_test = create_transformer_sequences(test_data, lookback)
        
        # 转换为PyTorch张量
        X_train_encoder = torch.FloatTensor(X_train_encoder)
        X_train_decoder = torch.FloatTensor(X_train_decoder)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_test_encoder = torch.FloatTensor(X_test_encoder)
        X_test_decoder = torch.FloatTensor(X_test_decoder)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_encoder, X_train_decoder, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_encoder, X_test_decoder, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = self.TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 1, dropout_prob)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型（简化版，没有早停）
        model.to(self.device)
        
        for epoch in range(min(epochs, 50)):  # Transformer训练较慢，限制轮数
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for encoder_input, decoder_input, y_batch in train_loader:
                encoder_input, decoder_input, y_batch = encoder_input.to(self.device), decoder_input.to(self.device), y_batch.to(self.device)
                
                # 前向传播
                outputs = model(encoder_input, decoder_input)
                loss = criterion(outputs, y_batch)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * encoder_input.size(0)
            
            train_loss /= len(train_loader.dataset)
        
        # 预测未来值
        forecasts = []
        
        # 使用最后一个lookback长度的序列作为初始编码器输入
        encoder_sequence = data_scaled[-lookback:].reshape(1, lookback, input_size)
        encoder_sequence_tensor = torch.FloatTensor(encoder_sequence).to(self.device)
        
        # 初始解码器输入：最后一个已知点
        decoder_input = data_scaled[-1:].reshape(1, 1, input_size)
        decoder_input_tensor = torch.FloatTensor(decoder_input).to(self.device)
        
        current_price = self.data[self.target_column].iloc[-1]
        
        for _ in range(forecast_steps):
            # 预测
            with torch.no_grad():
                model.eval()
                next_pred = model(encoder_sequence_tensor, decoder_input_tensor)
                next_pred_np = next_pred.cpu().numpy()[0, 0]
            
            # 转换为实际价格预测
            if input_size == 1:
                # 简单情况：只使用收益率作为输入
                pred_return = scaler.inverse_transform(np.array([[next_pred_np]]))[0, 0]
                pred_price = current_price * (1 + pred_return)
            else:
                # 复杂情况：使用完整特征矩阵
                # 为逆归一化创建一个完整的特征向量，只修改目标列(fund_close)
                # 目标列索引为0（根据create_transformer_sequences函数）
                last_features = data_scaled[-1:].copy()
                last_features[0, 0] = next_pred_np
                # 逆归一化整个特征向量
                last_features_inv = scaler.inverse_transform(last_features)
                # 获取逆归一化后的目标值
                pred_close = last_features_inv[0, 0]
                # 计算预测价格（假设我们预测的是下一个收盘价）
                pred_price = pred_close
            
            forecasts.append(pred_price)
            
            # 更新当前价格
            current_price = pred_price
            
            # 更新编码器序列（移除最早的点，添加最新预测）
            if input_size == 1:
                next_pred_reshaped = np.array([[next_pred_np]])
            else:
                # 更新完整特征向量中的目标列
                next_features = data_scaled[-1:].copy()
                next_features[0, 0] = next_pred_np
                next_pred_reshaped = next_features
            
            encoder_sequence = np.append(encoder_sequence[:, 1:, :], next_pred_reshaped.reshape(1, 1, input_size), axis=1)
            encoder_sequence_tensor = torch.FloatTensor(encoder_sequence).to(self.device)
            
            # 更新解码器输入为最新预测
            decoder_input = next_pred_reshaped.reshape(1, 1, input_size)
            decoder_input_tensor = torch.FloatTensor(decoder_input).to(self.device)
        
        self.models['transformer_pytorch'] = {
            'model': model,
            'params': {
                'lookback': lookback,
                'forecast_steps': forecast_steps,
                'd_model': d_model,
                'nhead': nhead,
                'num_encoder_layers': num_encoder_layers,
                'num_decoder_layers': num_decoder_layers,
                'dim_feedforward': dim_feedforward,
                'dropout_prob': dropout_prob,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        self.predictions['transformer_pytorch'] = forecasts
        
        logging.info("PyTorch Transformer模型训练完成")
        
        return forecasts
    
    def tft_pytorch(self, lookback=20, forecast_steps=1, hidden_size=64, num_heads=4, num_layers=2, dropout_prob=0.1, epochs=100, batch_size=32, learning_rate=0.001, feature_matrix=None):
        """PyTorch Temporal Fusion Transformer (TFT)预测模型"""
        logging.info("开始训练PyTorch TFT模型...")
        
        # 确定输入数据
        if feature_matrix is None:
            # 使用目标列的收益率作为输入
            data = self.data['return'].values.reshape(-1, 1)
            input_size = 1
        else:
            # 使用提供的特征矩阵
            data = feature_matrix.values
            input_size = data.shape[1]
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        self.scalers['tft_pytorch'] = scaler
        
        # 创建训练和测试序列
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        X_train, y_train = self.create_sequences(train_data, lookback, 1)
        X_test, y_test = self.create_sequences(test_data, lookback, 1)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = self.TFTModel(input_size, hidden_size, num_heads, num_layers, dropout_prob)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        model = self.train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience=10)
        
        # 预测未来值
        forecasts = []
        
        # 使用最后一个lookback长度的序列作为初始输入
        current_sequence = data_scaled[-lookback:].reshape(1, lookback, input_size)
        current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        for _ in range(forecast_steps):
            # 预测
            with torch.no_grad():
                model.eval()
                next_pred = model(current_sequence_tensor)
                next_pred_np = next_pred.cpu().numpy()[0, 0]
            
            forecasts.append(next_pred_np)
            
            # 更新序列
            next_pred_reshaped = np.array([[next_pred_np]]) if input_size == 1 else np.repeat([[next_pred_np]], input_size, axis=1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_pred_reshaped.reshape(1, 1, input_size), axis=1)
            current_sequence_tensor = torch.FloatTensor(current_sequence).to(self.device)
        
        # 逆归一化预测结果
        if input_size == 1:
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            forecasts_inv = scaler.inverse_transform(forecasts_reshaped).flatten()
        else:
            # 多变量情况，只逆归一化第一列（目标列）
            forecasts_reshaped = np.array(forecasts).reshape(-1, 1)
            # 创建全零数组，只填第一列
            full_forecasts = np.zeros((len(forecasts), input_size))
            full_forecasts[:, 0] = forecasts_reshaped.flatten()
            forecasts_inv = scaler.inverse_transform(full_forecasts)[:, 0]
        
        # 转换为实际价格预测
        last_price = self.data[self.target_column].iloc[-1]
        price_forecasts = []
        current_price = last_price
        
        for pred_return in forecasts_inv:
            pred_price = current_price * (1 + pred_return)
            price_forecasts.append(pred_price)
            current_price = pred_price
        
        self.models['tft_pytorch'] = {
            'model': model,
            'params': {
                'lookback': lookback,
                'forecast_steps': forecast_steps,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout_prob': dropout_prob,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
        
        self.predictions['tft_pytorch'] = price_forecasts
        
        logging.info("PyTorch TFT模型训练完成")
        
        return price_forecasts