# coding: utf-8

import pandas as pd
import numpy as np


# 1. Chia dữ liệu huấn luyện-kiểm tra-phân tích
def train_test_split(df, test_size, random_state=None):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(df))
    test_indices = np.random.choice(indices, size=test_size, replace=False)
    train_indices = np.setdiff1d(indices, test_indices)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    train_df = df.iloc[train_indices].reset_index(drop=True)
    
    return train_df, test_df


# 2. Phân biệt các đặc trưng danh từ và liên tục
def determine_type_of_feature(df, n_unique_values_treshold=15):
    
    feature_types = []
    for feature in df.columns:
        if feature != "label":
            dtype = df[feature].dtype
            n_unique = df[feature].nunique()
            if pd.api.types.is_object_dtype(dtype) or n_unique <= n_unique_values_treshold:
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# 3. Độ chính xác
def calculate_accuracy(predictions, labels):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return np.mean(predictions == labels)