import pickle
import os
import pandas as pd

# 设置缓存目录
cache_dir = 'data_cache'

# 遍历所有缓存文件
for file in os.listdir(cache_dir):
    if file.endswith('.pkl'):
        path = os.path.join(cache_dir, file)
        print(f'=== {file} ===')
        
        try:
            # 加载缓存数据
            data = pickle.load(open(path, 'rb'))
            
            # 检查数据类型
            if isinstance(data, pd.DataFrame):
                print(f'数据类型: DataFrame')
                print(f'数据形状: {data.shape}')
                print(f'列名: {list(data.columns)}')
                print('前3行数据:')
                print(data.head(3))
                print('')
            else:
                print(f'数据类型: {type(data)}')
                print(f'数据内容: {data}')
                print('')
        except Exception as e:
            print(f'加载缓存失败: {e}')
            print('')