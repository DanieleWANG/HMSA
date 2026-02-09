"""
预处理的时间序列数据集类
直接从预处理的 numpy array 加载，避免动态构建
"""
import pickle
import numpy as np
import pandas as pd


class PreprocessedTimeSeriesDataset:
    """从预处理的时间序列数据加载，高效且省内存"""
    def __init__(self, data_path):
        """
        Parameters
        ----------
        data_path : str
            预处理的时间序列数据文件路径 (.pkl)
        """
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']  # (N, T, F) numpy array
        self.index = data_dict['index']  # pd.MultiIndex
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """直接返回预计算的时间序列数据，非常快"""
        if isinstance(idx, (int, np.integer)):
            return self.data[idx]  # (T, F)
        elif isinstance(idx, (list, np.ndarray, tuple)):
            return self.data[idx]  # (N, T, F)
        else:
            return self.data[int(idx)]
    
    def get_index(self):
        """返回索引，兼容 qlib 数据集接口"""
        return self.index

