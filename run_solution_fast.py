"""
社交图谱链接预测解决方案 - 高性能版
优化预测速度，通过批处理、预计算和并行处理提高效率
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
import random
import warnings
from collections import defaultdict
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_json_data(file_path):
    """加载JSON数据"""
    print(f"正在加载数据: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {file_path}")):
            try:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                json_obj = json.loads(line)
                if isinstance(json_obj, list):
                    data.extend(json_obj)
                else:
                    data.append(json_obj)
            except Exception as e:
                print(f"解析第{i}行时出错: {e}, 内容: {line[:100]}")
    
    if not data:
        print(f"警告: 从 {file_path} 读取的数据为空!")
        return pd.DataFrame()
        
    result = pd.DataFrame(data)
    print(f"从 {file_path} 成功读取 {len(result)} 行数据")
    return result

def load_features():
    """加载已计算好的特征"""
    print("加载特征数据...")
    
    features = {}
    
    # 加载用户活跃度特征
    if os.path.exists('results/features/user_activity.csv'):
        features['user_activity'] = pd.read_csv('results/features/user_activity.csv')
        print(f"加载用户活跃度特征: {len(features['user_activity'])}行")
    else:
        print("用户活跃度特征文件不存在")
    
    # 加载商品流行度特征
    if os.path.exists('results/features/item_popularity.csv'):
        features['item_popularity'] = pd.read_csv('results/features/item_popularity.csv')
        print(f"加载商品流行度特征: {len(features['item_popularity'])}行")
    else:
        print("商品流行度特征文件不存在")
    
    # 加载网络特征
    if os.path.exists('results/features/network_features.csv'):
        features['network_features'] = pd.read_csv('results/features/network_features.csv')
        print(f"加载网络特征: {len(features['network_features'])}行")
    else:
        print("网络特征文件不存在")
    
    # 加载用户-商品交互频次
    if os.path.exists('results/features/user_item_freq.csv'):
        features['user_item_freq'] = pd.read_csv('results/features/user_item_freq.csv')
        print(f"加载用户-商品交互频次: {len(features['user_item_freq'])}行")
    else:
        print("用户-商品交互频次文件不存在")
    
    # 加载用户-用户交互频次
    if os.path.exists('results/features/user_user_freq.csv'):
        features['user_user_freq'] = pd.read_csv('results/features/user_user_freq.csv')
        print(f"加载用户-用户交互频次: {len(features['user_user_freq'])}行")
    else:
        print("用户-用户交互频次文件不存在")
    
    # 加载用户-类别交互频次
    if os.path.exists('results/features/user_cate_freq.csv'):
        features['user_cate_freq'] = pd.read_csv('results/features/user_cate_freq.csv')
        print(f"加载用户-类别交互频次: {len(features['user_cate_freq'])}行")
    else:
        print("用户-类别交互频次文件不存在")

    # 加载用户-一级类别交互频次
    if os.path.exists('results/features/user_cate1_freq.csv'):
        features['user_cate1_freq'] = pd.read_csv('results/features/user_cate1_freq.csv')
        print(f"加载用户-一级类别交互频次: {len(features['user_cate1_freq'])}行")
    else:
        print("用户-一级类别交互频次文件不存在")
    
    return features

def load_model():
    """加载训练好的模型"""
    print("加载模型...")
    
    model_path = 'results/models/lgb_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("模型加载成功")
        return model
    else:
        print("模型文件不存在")
        return None

def precompute_user_user_interactions(features):
    """预计算用户-用户交互历史，加速查找"""
    print("预计算用户-用户交互历史...")
    
    user_user_interactions = defaultdict(list)
    
    if 'user_user_freq' in features:
        # 直接使用DataFrame操作，避免行迭代
        user_user_df = features['user_user_freq']
        
        # 按inviter_id分组并转换为字典
        grouped = user_user_df.groupby('inviter_id')
        
        for inviter_id, group in tqdm(grouped, desc="预计算交互历史"):
            # 按交互频率排序
            sorted_group = group.sort_values('user_interaction_freq', ascending=False)
            # 转换为(voter_id, freq)元组列表
            interactions = list(zip(sorted_group['voter_id'], sorted_group['user_interaction_freq']))
            user_user_interactions[inviter_id] = interactions
    
    print(f"预计算完成，共有{len(user_user_interactions)}个用户的交互历史")
    return user_user_interactions

def precompute_feature_maps(features):
    """预计算特征映射，加速查找"""
    print("预计算特征映射...")
    
    feature_maps = {}
    
    # 用户活跃度特征
    if 'user_activity' in features:
        # 使用DataFrame操作代替行迭代
        user_activity_df = features['user_activity']
        
        # 创建邀请者特征字典
        inviter_activity = user_activity_df.set_index('user_id').to_dict('index')
        # 重命名键以匹配特征名称
        inviter_activity = {
            k: {'inviter_invite_count': v['invite_count'], 
                'inviter_voted_count': v['voted_count']}
            for k, v in inviter_activity.items()
        }
        
        # 创建回流者特征字典
        voter_activity = user_activity_df.set_index('user_id').to_dict('index')
        # 重命名键以匹配特征名称
        voter_activity = {
            k: {'voter_invite_count': v['invite_count'], 
                'voter_voted_count': v['voted_count']}
            for k, v in voter_activity.items()
        }
        
        feature_maps['inviter_activity'] = inviter_activity
        feature_maps['voter_activity'] = voter_activity
    
    # 商品流行度特征
    if 'item_popularity' in features:
        # 直接使用Series.to_dict()
        item_pop = features['item_popularity'].set_index('item_id')['item_popularity'].to_dict()
        feature_maps['item_popularity'] = item_pop
    
    # 网络特征
    if 'network_features' in features:
        # 使用DataFrame操作代替行迭代
        network_features_df = features['network_features']
        network_features = network_features_df.set_index('user_id').to_dict('index')
        feature_maps['network_features'] = network_features
    
    # 用户-商品交互频次
    if 'user_item_freq' in features:
        # 创建复合键
        user_item_df = features['user_item_freq']
        user_item_df['key'] = list(zip(user_item_df['inviter_id'], user_item_df['item_id']))
        user_item_freq = user_item_df.set_index('key')['interaction_freq'].to_dict()
        feature_maps['user_item_freq'] = user_item_freq
    
    # 用户-类别交互频次
    if 'user_cate_freq' in features:
        try:
            # 创建复合键
            user_cate_df = features['user_cate_freq']
            # 检查列名
            if 'inviter_id' in user_cate_df.columns and 'cate_id' in user_cate_df.columns:
                user_id_col = 'inviter_id'
            elif 'voter_id' in user_cate_df.columns and 'cate_id' in user_cate_df.columns:
                user_id_col = 'voter_id'
            else:
                # 如果找不到预期的列名，打印出实际的列名
                print(f"用户-类别交互频次文件的列名: {user_cate_df.columns.tolist()}")
                # 使用第一列作为用户ID列，第二列作为类别ID列
                if len(user_cate_df.columns) >= 2:
                    user_id_col = user_cate_df.columns[0]
                    cate_id_col = user_cate_df.columns[1]
                    print(f"尝试使用 {user_id_col} 作为用户ID列，{cate_id_col} 作为类别ID列")
                else:
                    raise ValueError(f"用户-类别交互频次文件的列数不足: {len(user_cate_df.columns)}")
                
            # 使用检测到的列名
            user_cate_df['key'] = list(zip(user_cate_df[user_id_col], user_cate_df['cate_id']))
            if 'interaction_freq' in user_cate_df.columns:
                freq_col = 'interaction_freq'
            else:
                # 尝试找到频率列
                numeric_cols = user_cate_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols and len(numeric_cols) > 0:
                    freq_col = numeric_cols[-1]  # 使用最后一个数值列作为频率列
                    print(f"尝试使用 {freq_col} 作为交互频率列")
                else:
                    raise ValueError("找不到合适的交互频率列")
                
            user_cate_freq = user_cate_df.set_index('key')[freq_col].to_dict()
            feature_maps['user_cate_freq'] = user_cate_freq
            print(f"成功预计算用户-类别交互频次，使用列: {user_id_col}, cate_id, {freq_col}")
        except Exception as e:
            print(f"预计算用户-类别交互频次失败: {e}")
            # 尝试打印前几行数据，帮助诊断问题
            try:
                print("用户-类别交互频次文件前5行数据预览:")
                print(features['user_cate_freq'].head())
            except:
                print("无法预览用户-类别交互频次数据")

    # 用户-一级类别交互频次
    if 'user_cate1_freq' in features:
        try:
            # 创建复合键
            user_cate1_df = features['user_cate1_freq']
            user_id_col = None
            cate1_id_col = None # 确保初始化

            actual_columns = user_cate1_df.columns.tolist()
            
            # 明确地检测 user_id 和 cate1_id 的列名
            if 'inviter_id' in actual_columns and 'cate1_id' in actual_columns:
                user_id_col = 'inviter_id'
                cate1_id_col = 'cate1_id'
                print(f"用户-一级类别交互频次文件检测到列: inviter_id, cate1_id")
            elif 'voter_id' in actual_columns and 'cate1_id' in actual_columns:
                user_id_col = 'voter_id'
                cate1_id_col = 'cate1_id'
                print(f"用户-一级类别交互频次文件检测到列: voter_id, cate1_id")
            elif 'inviter_id' in actual_columns and 'cate_level1_id' in actual_columns:
                user_id_col = 'inviter_id'
                cate1_id_col = 'cate_level1_id'
                print(f"用户-一级类别交互频次文件检测到列: inviter_id, cate_level1_id")
            elif 'voter_id' in actual_columns and 'cate_level1_id' in actual_columns:
                user_id_col = 'voter_id'
                cate1_id_col = 'cate_level1_id'
                print(f"用户-一级类别交互频次文件检测到列: voter_id, cate_level1_id")
            elif len(actual_columns) >= 2: # 回退到基于位置的列名获取
                print(f"用户-一级类别交互频次文件回退检测列名: {actual_columns}")
                user_id_col = actual_columns[0]
                cate1_id_col = actual_columns[1]
                print(f"尝试使用 {user_id_col} 作为用户ID列，{cate1_id_col} 作为一级类别ID列")
            else:
                raise ValueError(f"用户-一级类别交互频次文件的列数不足: {len(actual_columns)}")

            if not user_id_col or not cate1_id_col:
                 raise ValueError(f"未能从列 {actual_columns} 中确定 user_id_col 或 cate1_id_col")
                
            # 使用检测到的列名创建复合键
            user_cate1_df['key'] = list(zip(user_cate1_df[user_id_col], user_cate1_df[cate1_id_col]))
            
            if 'interaction_freq' in user_cate1_df.columns:
                freq_col = 'interaction_freq'
            else:
                # 尝试找到频率列
                numeric_cols = user_cate1_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols and len(numeric_cols) > 0:
                    freq_col = numeric_cols[-1]  # 使用最后一个数值列作为频率列
                    print(f"尝试使用 {freq_col} 作为交互频率列")
                else:
                    raise ValueError("找不到合适的交互频率列")
                
            user_cate1_freq = user_cate1_df.set_index('key')[freq_col].to_dict()
            feature_maps['user_cate1_freq'] = user_cate1_freq
            print(f"成功预计算用户-一级类别交互频次，使用列: {user_id_col}, cate1_id, {freq_col}")
        except Exception as e:
            print(f"预计算用户-一级类别交互频次失败: {e}")
            # 尝试打印前几行数据，帮助诊断问题
            try:
                print("用户-一级类别交互频次文件前5行数据预览:")
                print(features['user_cate1_freq'].head())
            except:
                print("无法预览用户-一级类别交互频次数据")
    
    print("预计算特征映射完成")
    return feature_maps

def get_features_for_voter(inviter_id, item_id, voter_id, feature_maps, user_user_interactions, item_cate_map=None):
    """为单个候选回流者构建特征"""
    features = {}
    
    # 添加邀请者活跃度特征
    if 'inviter_activity' in feature_maps and inviter_id in feature_maps['inviter_activity']:
        features.update(feature_maps['inviter_activity'][inviter_id])
    else:
        features['inviter_invite_count'] = 0
        features['inviter_voted_count'] = 0
    
    # 添加回流者活跃度特征
    if 'voter_activity' in feature_maps and voter_id in feature_maps['voter_activity']:
        features.update(feature_maps['voter_activity'][voter_id])
    else:
        features['voter_invite_count'] = 0
        features['voter_voted_count'] = 0
    
    # 添加商品流行度特征
    if 'item_popularity' in feature_maps and item_id in feature_maps['item_popularity']:
        features['item_popularity'] = feature_maps['item_popularity'][item_id]
    else:
        features['item_popularity'] = 0
    
    # 添加网络特征 - 邀请者
    if 'network_features' in feature_maps and inviter_id in feature_maps['network_features']:
        features['inviter_in_degree'] = feature_maps['network_features'][inviter_id]['in_degree']
        features['inviter_out_degree'] = feature_maps['network_features'][inviter_id]['out_degree']
    else:
        features['inviter_in_degree'] = 0
        features['inviter_out_degree'] = 0
    
    # 添加网络特征 - 回流者
    if 'network_features' in feature_maps and voter_id in feature_maps['network_features']:
        features['voter_in_degree'] = feature_maps['network_features'][voter_id]['in_degree']
        features['voter_out_degree'] = feature_maps['network_features'][voter_id]['out_degree']
    else:
        features['voter_in_degree'] = 0
        features['voter_out_degree'] = 0
    
    # 添加用户-商品交互频率
    if 'user_item_freq' in feature_maps and (inviter_id, item_id) in feature_maps['user_item_freq']:
        features['inviter_item_freq'] = feature_maps['user_item_freq'][(inviter_id, item_id)]
    else:
        features['inviter_item_freq'] = 0
    
    # 添加回流者-商品交互频率
    if 'user_item_freq' in feature_maps and (voter_id, item_id) in feature_maps['user_item_freq']:
        features['voter_item_freq'] = feature_maps['user_item_freq'][(voter_id, item_id)]
    else:
        features['voter_item_freq'] = 0
    
    # 添加用户-用户交互频率 - 使用二分查找优化
    features['user_interaction_freq'] = 0
    if inviter_id in user_user_interactions:
        interactions = user_user_interactions[inviter_id]
        # 二分查找
        left, right = 0, len(interactions) - 1
        while left <= right:
            mid = (left + right) // 2
            if interactions[mid][0] == voter_id:
                features['user_interaction_freq'] = interactions[mid][1]
                break
            elif interactions[mid][0] < voter_id:
                left = mid + 1
            else:
                right = mid - 1
    
    # 添加类别相关特征
    cate_id = None
    cate1_id = None
    
    if item_cate_map and item_id in item_cate_map:
        cate_id = item_cate_map[item_id].get('cate_id')
        cate1_id = item_cate_map[item_id].get('cate1_id')
    
    # 添加类别特征，使用安全的键值检查
    features['voter_cate_freq'] = 0
    features['inviter_cate_freq'] = 0
    
    if cate_id is not None:
        # 用户-类别交互频率 - 安全地尝试不同的键格式
        if 'user_cate_freq' in feature_maps:
            # 尝试多种可能的键格式
            for key in [(voter_id, cate_id), (str(voter_id), str(cate_id))]:
                if key in feature_maps['user_cate_freq']:
                    features['voter_cate_freq'] = feature_maps['user_cate_freq'][key]
                    break
                    
            for key in [(inviter_id, cate_id), (str(inviter_id), str(cate_id))]:
                if key in feature_maps['user_cate_freq']:
                    features['inviter_cate_freq'] = feature_maps['user_cate_freq'][key]
                    break
    
    # 计算特征比例和差值特征
    # 交互频率比例
    if features['voter_item_freq'] > 0 and features['inviter_item_freq'] > 0:
        features['item_freq_ratio'] = features['voter_item_freq'] / features['inviter_item_freq']
    else:
        features['item_freq_ratio'] = 0
        
    # 邀请者和回流者的活跃度比例
    if features['voter_voted_count'] > 0 and features['inviter_voted_count'] > 0:
        features['voted_count_ratio'] = features['voter_voted_count'] / features['inviter_voted_count']
    else:
        features['voted_count_ratio'] = 0
        
    if features['voter_invite_count'] > 0 and features['inviter_invite_count'] > 0:
        features['invite_count_ratio'] = features['voter_invite_count'] / features['inviter_invite_count']
    else:
        features['invite_count_ratio'] = 0
    
    # 社交网络度数比例
    if features['voter_in_degree'] > 0 and features['inviter_in_degree'] > 0:
        features['in_degree_ratio'] = features['voter_in_degree'] / features['inviter_in_degree']
    else:
        features['in_degree_ratio'] = 0
        
    if features['voter_out_degree'] > 0 and features['inviter_out_degree'] > 0:
        features['out_degree_ratio'] = features['voter_out_degree'] / features['inviter_out_degree']
    else:
        features['out_degree_ratio'] = 0
    
    # 计算交互强度 - 综合考虑多种交互频率
    features['interaction_strength'] = (
        features['user_interaction_freq'] * 5 +  # 直接交互权重最高
        features['voter_item_freq'] * 2 +        # 回流者与商品的交互次数
        features['voter_cate_freq'] * 1          # 回流者与类别的交互次数
    )
    
    # 社交影响力
    features['social_influence'] = features['voter_in_degree'] * features['voter_out_degree']
    features['inviter_influence'] = features['inviter_in_degree'] * features['inviter_out_degree']
    
    return features

def process_batch(batch_data, feature_maps, user_user_interactions, model, all_voters, item_cate_map=None):
    """处理一批测试数据"""
    # 定义特征列 - 使用全部24个特征
    feature_cols = [
        'user_interaction_freq',
        'interaction_strength',
        'voter_voted_count',
        'inviter_out_degree',
        'inviter_invite_count',
        'voter_cate_freq',
        'item_popularity',
        'inviter_cate_freq',
        'inviter_item_freq',
        'inviter_voted_count'
    ]
    
    # model_feature_cols 现在应该与 feature_cols 一致，因为我们假设模型将用所有这些特征重新训练
    model_feature_cols = feature_cols 
    
    batch_predictions = []
    
    for _, row in batch_data.iterrows():
        inviter_id = row['inviter_id']
        item_id = row['item_id']
        triple_id = row['triple_id']
        
        # 获取候选回流者 - 优化候选回流者筛选策略
        candidates = []
        
        # 1. 优先添加历史交互频繁的用户
        if inviter_id in user_user_interactions and len(user_user_interactions[inviter_id]) > 0:
            # 使用历史交互中最常见的回流者，最多取25个
            top_historical_voters = [voter_id for voter_id, _ in user_user_interactions[inviter_id][:25]]
            candidates.extend(top_historical_voters)
        
        # 2. 添加与该商品交互过的用户
        if 'user_item_freq' in feature_maps:
            item_interacted_users = []
            # 查找所有与该商品交互过的用户
            for (user_id, item), freq in feature_maps['user_item_freq'].items():
                if item == item_id and user_id != inviter_id and user_id not in candidates:
                    item_interacted_users.append((user_id, freq))
            
            # 按交互频率排序并选择前15个
            item_interacted_users.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([user_id for user_id, _ in item_interacted_users[:15]])
        
        # 3. 添加与相同类别交互过的用户
        if item_cate_map and item_id in item_cate_map and 'user_cate_freq' in feature_maps:
            cate_id = item_cate_map[item_id]['cate_id']
            cate_interacted_users = []
            
            # 查找所有与该类别交互过的用户
            for (user_id, cate), freq in feature_maps['user_cate_freq'].items():
                if cate == cate_id and user_id != inviter_id and user_id not in candidates:
                    cate_interacted_users.append((user_id, freq))
            
            # 按交互频率排序并选择前10个
            cate_interacted_users.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([user_id for user_id, _ in cate_interacted_users[:10]])
        
        # 4. 添加网络度数高的用户（具有社交影响力的用户）
        if 'network_features' in feature_maps:
            influential_users = []
            
            for user_id, features in feature_maps['network_features'].items():
                if user_id != inviter_id and user_id not in candidates:
                    # 计算社交影响力指数
                    influence = features['in_degree'] * features['out_degree']
                    influential_users.append((user_id, influence))
            
            # 按影响力排序并选择前10个
            influential_users.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([user_id for user_id, _ in influential_users[:10]])
        
        # 5. 如果候选人数不足，添加活跃度高的用户
        if len(candidates) < 50 and 'user_activity' in feature_maps:
            active_users = []
            
            for user_id in feature_maps['voter_activity']:
                if user_id != inviter_id and user_id not in candidates:
                    activity = feature_maps['voter_activity'][user_id]['voter_voted_count'] + feature_maps['voter_activity'][user_id]['voter_invite_count']
                    active_users.append((user_id, activity))
            
            # 按活跃度排序并选择足够的用户
            active_users.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([user_id for user_id, _ in active_users[:min(50 - len(candidates), len(active_users))]])
        
        # 6. 如果仍然不足，随机添加
        if len(candidates) < 50:
            # 排除已有的候选回流者
            remaining_voters = [v for v in all_voters if v not in candidates]
            # 随机选择一些回流者
            random_voters = np.random.choice(remaining_voters, min(50 - len(candidates), len(remaining_voters)), replace=False)
            candidates.extend(random_voters.tolist())
        
        # 确保候选人数量不超过100
        candidates = candidates[:100]
        
        # 为每个候选回流者构建特征
        candidate_features = []
        
        for voter_id in candidates:
            # 构建特征字典
            features = get_features_for_voter(inviter_id, item_id, voter_id, feature_maps, user_user_interactions, item_cate_map)
            candidate_features.append(features)
        
        # 转换为DataFrame
        candidates_df = pd.DataFrame(candidate_features)
        
        # 确保所有特征列都存在
        for model_col in model_feature_cols:
            if model_col not in candidates_df.columns:
                candidates_df[model_col] = 0
        
        candidates_df_for_model = candidates_df[model_feature_cols]
        
        # 使用模型预测概率
        try:
            # 预测概率
            probs = model.predict_proba(candidates_df_for_model)[:, 1]
            
            # 将概率与候选回流者ID配对
            voter_probs = list(zip(candidates, probs))
            
            # 按概率排序
            voter_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 选择概率最高的5个回流者
            top_voters = [str(voter_id) for voter_id, _ in voter_probs[:5]]
            
            # 确保有5个推荐
            if len(top_voters) < 5:
                # 如果不足5个，随机添加一些用户
                remaining = [str(v) for v in all_voters if str(v) not in top_voters]
                additional = random.sample(remaining, min(5 - len(top_voters), len(remaining)))
                top_voters.extend(additional)
            
            batch_predictions.append({
                'triple_id': str(triple_id),
                'candidate_voter_list': top_voters[:5]
            })
        except Exception as e:
            print(f"预测样本 {triple_id} 失败: {e}")
            # 如果预测失败，随机推荐5个用户
            random_voters = [str(v) for v in np.random.choice(all_voters, 5, replace=False)]
            batch_predictions.append({
                'triple_id': str(triple_id),
                'candidate_voter_list': random_voters
            })
    
    return batch_predictions

def load_item_cate_map():
    """加载商品与类别的映射关系"""
    print("加载商品与类别的映射关系...")
    
    item_cate_map = {}
    
    # 尝试从商品信息文件加载
    try:
        if os.path.exists('data/train/item_info.json'):
            with open('data/train/item_info.json', 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="读取商品信息"):
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, list):
                            for item in data:
                                if 'item_id' in item and 'cate_id' in item:
                                    item_cate_map[item['item_id']] = {
                                        'cate_id': item['cate_id'],
                                        'cate1_id': item.get('cate1_id', item['cate_id'])  # 如果没有一级类别，使用类别ID
                                    }
                        else:
                            if 'item_id' in data and 'cate_id' in data:
                                item_cate_map[data['item_id']] = {
                                    'cate_id': data['cate_id'],
                                    'cate1_id': data.get('cate1_id', data['cate_id'])  # 如果没有一级类别，使用类别ID
                                }
                    except:
                        continue
            
            print(f"成功加载 {len(item_cate_map)} 个商品的类别信息")
            return item_cate_map
    except Exception as e:
        print(f"加载商品类别信息失败: {e}")
    
    # 如果无法加载，返回空字典
    return {}

def generate_predictions_parallel(test_data, model, feature_maps, user_user_interactions, all_voters):
    """
    使用joblib并行生成预测
    
    参数:
        test_data: 测试数据
        model: 训练好的模型
        feature_maps: 预计算的特征映射
        user_user_interactions: 预计算的用户-用户交互历史
        all_voters: 所有可能的回流者
        
    返回:
        预测结果
    """
    print("为测试数据生成预测（并行版）...")
    
    # 加载商品与类别的映射关系
    item_cate_map = load_item_cate_map()
    
    # 批处理大小
    batch_size = 200
    
    # 将测试数据分成多个批次
    batches = []
    for i in range(0, len(test_data), batch_size):
        batches.append(test_data.iloc[i:i+batch_size])
    
    print(f"将数据分成 {len(batches)} 个批次进行处理")
    
    # 使用joblib并行处理
    n_jobs = min(os.cpu_count(), 8)  # 限制最大进程数
    print(f"使用 {n_jobs} 个并行作业处理")
    
    try:
        # 使用joblib并行处理
        all_batch_predictions = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_batch)(batch, feature_maps, user_user_interactions, model, all_voters, item_cate_map)
            for batch in batches
        )
        
        # 合并所有批次的预测结果
        predictions = []
        for batch_predictions in all_batch_predictions:
            predictions.extend(batch_predictions)
        
        print(f"预测完成，共生成 {len(predictions)} 个预测结果")
        return predictions
    except Exception as e:
        print(f"并行处理失败: {e}")
        print("回退到串行处理...")
        
        # 回退到串行处理
        predictions = []
        for i, batch in enumerate(tqdm(batches, desc="生成预测（串行处理）")):
            batch_predictions = process_batch(batch, feature_maps, user_user_interactions, model, all_voters, item_cate_map)
            predictions.extend(batch_predictions)
        
        print(f"预测完成，共生成 {len(predictions)} 个预测结果")
        return predictions

def save_predictions(predictions, output_file='results/output/predictions_parallel.json'):
    """保存预测结果"""
    print(f"保存预测结果到 {output_file}...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False)
    
    print("预测结果已保存")

def calculate_mrr(predictions, validation_data):
    """
    计算MRR@5指标
    
    参数:
        predictions: 预测结果列表，每个元素包含triple_id和candidate_voter_list
        validation_data: 验证数据，包含真实的回流者信息
        
    返回:
        MRR@5值
    """
    print("计算MRR@5指标...")
    
    # 将预测结果转换为字典，方便查找
    pred_dict = {pred['triple_id']: pred['candidate_voter_list'] for pred in predictions}
    
    # 初始化变量
    total_queries = 0
    total_reciprocal_rank = 0
    hits_count = 0
    
    # 遍历验证数据
    for _, row in tqdm(validation_data.iterrows(), desc="计算MRR", total=len(validation_data)):
        triple_id = str(row['triple_id'])
        true_voter_id = str(row['voter_id'])
        
        # 如果该triple_id有预测结果
        if triple_id in pred_dict:
            total_queries += 1
            candidate_list = pred_dict[triple_id]
            
            # 查找真实回流者在候选列表中的位置
            if true_voter_id in candidate_list:
                rank = candidate_list.index(true_voter_id) + 1  # 索引从0开始，排名从1开始
                total_reciprocal_rank += 1.0 / rank
                hits_count += 1
    
    # 计算MRR@5
    mrr = total_reciprocal_rank / total_queries if total_queries > 0 else 0
    hits_rate = hits_count / total_queries if total_queries > 0 else 0
    
    print(f"评估结果 - MRR@5: {mrr:.5f}, HITS@5: {hits_rate:.5f}, 总查询数: {total_queries}")
    return mrr, hits_rate

def main():
    """主函数，执行预测流程"""
    # 确保目录存在
    os.makedirs('results/output', exist_ok=True)
    
    # 加载测试数据
    test_data = load_json_data('data/item_share_preliminary_test_info.json')
    
    # 添加triple_id到测试集
    test_data['triple_id'] = test_data.index.astype(str)
    
    # 加载用户信息
    user_info = load_json_data('data/user_info.json')
    
    # 获取所有可能的回流者
    all_voters = user_info['user_id'].unique()
    
    # 加载特征
    features = load_features()
    
    # 预计算用户-用户交互历史
    user_user_interactions = precompute_user_user_interactions(features)
    
    # 预计算特征映射
    feature_maps = precompute_feature_maps(features)
    
    # 加载模型
    model = load_model()
    
    if model is None:
        print("模型加载失败，无法继续预测")
        return
    
    # 生成预测
    predictions = generate_predictions_parallel(test_data, model, feature_maps, user_user_interactions, all_voters)
    
    # 保存预测结果
    save_predictions(predictions)
    
    # 如果有验证集，计算MRR
    # validation_file = 'data/test/item_share_preliminary_test_info.json'
    validation_file = 'data/train/item_share_train_info.json'
    if os.path.exists(validation_file):
        validation_data = load_json_data(validation_file)
        mrr, hits_rate = calculate_mrr(predictions, validation_data)
        
        # 将评估结果保存到文件
        with open('results/output/evaluation_results.txt', 'w') as f:
            f.write(f"MRR@5: {mrr:.5f}\n")
            f.write(f"HITS@5: {hits_rate:.5f}\n")
    
    print("预测流程执行完成")

if __name__ == "__main__":
    main() 