# decision_tree_functions.py
# coding: utf-8

import numpy as np
import pandas as pd
import random

from helper_functions import determine_type_of_feature

# Khởi tạo biến toàn cục an toàn
COLUMN_HEADERS = None
FEATURE_TYPES = None

# 1. Cây quyết định hỗ trợ
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    return len(unique_classes) == 1

def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    return unique_classes[index]

def get_potential_splits(data, random_subspace, feature_types):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))  # excluding the last column (label)
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    
    for column_index in column_indices:
        values = data[:, column_index]
        if feature_types[column_index] == "continuous":
            unique_values = np.unique(values)
            # Đề xuất split tại trung điểm giữa các giá trị liên tiếp
            potential_splits[column_index] = (unique_values[:-1] + unique_values[1:]) / 2
        else:
            potential_splits[column_index] = np.unique(values)
    
    return potential_splits

def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    # Sử dụng np.where để tránh log2(0)
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))
    return entropy

def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    if n == 0:
        return 0
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    return overall_entropy

def determine_best_split(data, potential_splits, feature_types):
    overall_entropy = float('inf')
    best_split_column = None
    best_split_value = None
    
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, column_index, value, feature_types)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if current_overall_entropy < overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def split_data(data, split_column, split_value, feature_types):
    split_column_values = data[:, split_column]
    type_of_feature = feature_types[split_column]
    
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

# 2. Thuật toán cây quyết định
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None, column_headers=None, feature_types=None):
    # Chuẩn bị dữ liệu
    if counter == 0:
        column_headers = df.columns
        feature_types = determine_type_of_feature(df)
        data = df.values
    else:
        data = df
    
    # Trường hợp cơ bản
    if (check_purity(data) or len(data) < min_samples or counter == max_depth):
        return classify_data(data)
    
    # Phần đệ quy
    counter += 1
    potential_splits = get_potential_splits(data, random_subspace, feature_types)
    split_column, split_value = determine_best_split(data, potential_splits, feature_types)
    
    if split_column is None:  # Nếu không tìm thấy phân chia hợp lệ
        return classify_data(data)
        
    data_below, data_above = split_data(data, split_column, split_value, feature_types)
    
    # Kiểm tra dữ liệu trống
    if len(data_below) == 0 or len(data_above) == 0:
        return classify_data(data)
    
    # Xác định câu hỏi
    feature_name = column_headers[split_column]
    type_of_feature = feature_types[split_column]
    question = f"{feature_name} <= {split_value}" if type_of_feature == "continuous" else f"{feature_name} = {split_value}"
    
    # Khởi tạo cây con
    sub_tree = {question: []}
    yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace, column_headers, feature_types)
    no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace, column_headers, feature_types)
    
    if yes_answer == no_answer:
        sub_tree = yes_answer
    else:
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
    
    return sub_tree

# 3. Dự đoán
def predict_example(example, tree):
    if not isinstance(tree, dict):
        return tree
    
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    return predict_example(example, answer)

def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions