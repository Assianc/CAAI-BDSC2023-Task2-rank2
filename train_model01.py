import os
import pandas as pd
import numpy as np
import pickle
import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import random
import datetime
import time
import sys


# 创建日志函数
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# 创建结果目录
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/features', exist_ok=True)  # 确保特征目录也存在

log_message("加载原始特征数据...")
try:
    user_activity = pd.read_csv('results/features/user_activity.csv')
    item_popularity = pd.read_csv('results/features/item_popularity.csv')
    network_features = pd.read_csv('results/features/network_features.csv')
    user_item_freq_orig = pd.read_csv('results/features/user_item_freq.csv')
    user_cate_freq_orig = pd.read_csv('results/features/user_cate_freq.csv')
    user_cate1_freq_orig = pd.read_csv('results/features/user_cate1_freq.csv')
    # 新增：加载用户-用户交互频率
    if os.path.exists('results/features/user_user_freq.csv'):
        user_user_freq = pd.read_csv('results/features/user_user_freq.csv')
        log_message("加载 user_user_freq.csv")
    else:
        user_user_freq = pd.DataFrame(columns=['inviter_id', 'voter_id', 'user_interaction_freq'])
        log_message("警告: results/features/user_user_freq.csv 未找到，user_interaction_freq 将为0。")
except FileNotFoundError as e:
    log_message(f"错误：一个或多个特征文件未找到: {e}")
    log_message("请确保所有必要的特征文件都存在于 'results/features/' 目录下。")
    sys.exit(1)


# 读取训练数据和辅助信息
def read_json_to_df(file_path, chunk_size=None, desc_override=None):
    desc_to_use = desc_override if desc_override else f"Reading {file_path}"
    lines_read = 0
    lines_parsed_successfully = 0
    if chunk_size:
        chunks = []
        dfs_yielded = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=desc_to_use)):
                lines_read += 1
                try:
                    line_content = line.strip()
                    if not line_content: continue
                    json_obj = json.loads(line_content)
                    if isinstance(json_obj, list):
                        chunks.extend(json_obj)
                        lines_parsed_successfully += len(json_obj)
                    else:
                        chunks.append(json_obj)
                        lines_parsed_successfully += 1
                    if (i + 1) % chunk_size == 0 and chunks:
                        if chunks:
                            yield pd.DataFrame(chunks)
                            dfs_yielded += 1
                        chunks = []
                except Exception as e:
                    log_message(f"Error parsing line {lines_read} in {file_path}: {e}, content: {line_content[:100]}")
            if chunks:
                yield pd.DataFrame(chunks)
                dfs_yielded += 1
        if dfs_yielded == 0:
            log_message(
                f"警告: read_json_to_df (chunked) for {file_path} did not yield any DataFrames. Total lines read: {lines_read}, lines parsed: {lines_parsed_successfully}")
            if lines_parsed_successfully == 0:
                return
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=desc_to_use)):
                lines_read += 1
                try:
                    line_content = line.strip()
                    if not line_content: continue
                    json_obj = json.loads(line_content)
                    if isinstance(json_obj, list):
                        data.extend(json_obj)
                        lines_parsed_successfully += len(json_obj)
                    else:
                        data.append(json_obj)
                        lines_parsed_successfully += 1
                except Exception as e:
                    log_message(f"Error parsing line {lines_read} in {file_path}: {e}, content: {line_content[:100]}")
        log_message(
            f"Finished reading {file_path}. Total lines read: {lines_read}, objects parsed: {lines_parsed_successfully}")
        if not data:
            log_message(f"警告: 从 {file_path} 读取的数据为空或未能解析任何对象! 将 yield 空 DataFrame.")
            yield pd.DataFrame()
        else:
            yield pd.DataFrame(data)


log_message("读取核心训练数据...")
try:
    train_data_chunks = list(
        read_json_to_df('data/item_share_train_info.json', chunk_size=200000, desc_override="Reading train_info"))
    if not train_data_chunks or all(df.empty for df in train_data_chunks):
        log_message("错误: train_data 未能加载或所有块均为空。")
        train_data = pd.DataFrame()
    else:
        train_data = pd.concat(train_data_chunks)
    if train_data.empty:
        log_message("警告: train_data 为空。")
except Exception as e:
    log_message(f"读取训练数据时发生错误: {e}")
    sys.exit(1)

log_message("读取用户信息...")
try:
    # 由于 read_json_to_df 总是返回生成器，我们需要list()它
    user_info_chunks = list(read_json_to_df('data/user_info.json', desc_override="Reading user_info"))
    if not user_info_chunks or user_info_chunks[0].empty:  # 期望非分块只产生一个块
        log_message(f"错误: user_info未能成功加载为DataFrame或为空。请检查 data/train/user_info.json 文件。")
        user_info = pd.DataFrame()  # 创建空df以避免后续None错误，但可能导致其他问题
        # exit() # 考虑是否应该在这里退出
    else:
        user_info = user_info_chunks[0]  # 非分块加载应该只有一个DataFrame

    if not isinstance(user_info, pd.DataFrame):  # 双重保险检查
        log_message(f"致命错误: user_info 最终不是 DataFrame。类型: {type(user_info)}")
        sys.exit(1)

    if user_info.empty:
        log_message("错误: user_info DataFrame 为空。")
        sys.exit(1)  # 如果为空，后续 'user_id' 检查会失败，不如早点退出

    if 'user_id' not in user_info.columns:
        log_message(
            f"错误: user_info DataFrame ({user_info.shape}行) 中缺少 'user_id' 列。可用列: {user_info.columns.tolist()}")
        sys.exit(1)
except Exception as e:
    log_message(f"读取用户信息时发生错误: {e}")
    sys.exit(1)

log_message("读取物品信息...")
try:
    item_info_chunks = list(
        read_json_to_df('data/item_info.json', chunk_size=100000, desc_override="Reading item_info"))
    if not item_info_chunks or all(df.empty for df in item_info_chunks):
        log_message("警告: item_info 未能加载或所有块均为空。将使用空DataFrame。")
        item_info = pd.DataFrame()
    else:
        item_info = pd.concat(item_info_chunks)

    if item_info.empty:
        log_message("警告: item_info 为空。")
except Exception as e:
    log_message(f"读取物品信息时发生错误: {e}")
    # 物品信息不是必须的，所以不退出
    item_info = pd.DataFrame()

# 转换时间格式 (如果需要)
# train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

print("构建训练样本...")
# 构建正样本（实际的 inviter-voter 对）
positive_samples = train_data[['inviter_id', 'voter_id', 'item_id']].copy()  # 添加 .copy() 避免 SettingWithCopyWarning
positive_samples['label'] = 1


# 构建负样本（随机选择 voter）
def generate_negative_samples(positive_df, all_users_df, ratio=1, random_seed=42):
    np.random.seed(random_seed)
    negative_samples_list = []
    all_possible_voters = all_users_df['user_id'].unique()

    for _, row in tqdm(positive_df.iterrows(), total=len(positive_df), desc="Generating negative samples"):
        inviter_id = row['inviter_id']
        item_id = row['item_id']
        true_voter_id = row['voter_id']

        num_neg_samples = 0
        attempts = 0
        max_attempts = len(all_possible_voters) * 2  # 避免无限循环

        while num_neg_samples < ratio and attempts < max_attempts:
            random_voter = np.random.choice(all_possible_voters)
            attempts += 1
            # 确保随机选择的 voter 不是原始的 voter，并且不是邀请者本身
            if random_voter != true_voter_id and random_voter != inviter_id:
                negative_samples_list.append({
                    'inviter_id': inviter_id,
                    'voter_id': random_voter,
                    'item_id': item_id,
                    'label': 0
                })
                num_neg_samples += 1

    return pd.DataFrame(negative_samples_list)


print("生成负样本...")
# 使用 user_info 中的 user_id 作为所有可能的回流者
negative_samples = generate_negative_samples(positive_samples, user_info, ratio=2)  # 增加负采样比例

# 合并正负样本
train_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
print(f"训练样本总数: {len(train_samples)}, 正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")

# --- 特征工程 ---
print("开始特征工程...")

# 1. 合并基础特征
# 用户活跃度 (inviter 和 voter)
train_samples = train_samples.merge(user_activity.rename(columns={
    'user_id': 'inviter_id', 'invite_count': 'inviter_invite_count', 'voted_count': 'inviter_voted_count'
}), on='inviter_id', how='left')
train_samples = train_samples.merge(user_activity.rename(columns={
    'user_id': 'voter_id', 'invite_count': 'voter_invite_count', 'voted_count': 'voter_voted_count'
}), on='voter_id', how='left')

# 商品流行度
train_samples = train_samples.merge(item_popularity, on='item_id', how='left')

# 网络特征 (inviter 和 voter)
network_cols_map_inviter = {'user_id': 'inviter_id', 'in_degree': 'inviter_in_degree',
                            'out_degree': 'inviter_out_degree'}
network_cols_map_voter = {'user_id': 'voter_id', 'in_degree': 'voter_in_degree', 'out_degree': 'voter_out_degree'}
train_samples = train_samples.merge(
    network_features[[col for col in network_cols_map_inviter.keys() if col in network_features.columns]].rename(
        columns=network_cols_map_inviter), on='inviter_id', how='left')
train_samples = train_samples.merge(
    network_features[[col for col in network_cols_map_voter.keys() if col in network_features.columns]].rename(
        columns=network_cols_map_voter), on='voter_id', how='left')

# 2. 合并交互频率特征
# inviter_item_freq (邀请者与当前物品的交互频率)
if 'inviter_id' in user_item_freq_orig.columns and 'item_id' in user_item_freq_orig.columns and 'interaction_freq' in user_item_freq_orig.columns:
    train_samples = train_samples.merge(
        user_item_freq_orig.rename(columns={'interaction_freq': 'inviter_item_freq'}),
        on=['inviter_id', 'item_id'], how='left'
    )
else:
    print(
        "警告: user_item_freq_orig 文件列名不符合预期 ('inviter_id', 'item_id', 'interaction_freq')，inviter_item_freq 可能不正确。")
    train_samples['inviter_item_freq'] = 0

# user_interaction_freq (邀请者与候选回流者的直接交互频率)
if 'inviter_id' in user_user_freq.columns and 'voter_id' in user_user_freq.columns and 'user_interaction_freq' in user_user_freq.columns:
    train_samples = train_samples.merge(
        user_user_freq,
        on=['inviter_id', 'voter_id'], how='left'
    )
else:
    print(
        "警告: user_user_freq 文件列名不符合预期 ('inviter_id', 'voter_id', 'user_interaction_freq')，user_interaction_freq 可能不正确。")
    train_samples['user_interaction_freq'] = 0

# voter_item_freq (回流者与当前物品的交互频率)
if 'inviter_id' in user_item_freq_orig.columns and 'item_id' in user_item_freq_orig.columns and 'interaction_freq' in user_item_freq_orig.columns:
    voter_item_freq_df = user_item_freq_orig.rename(
        columns={'inviter_id': 'voter_id', 'interaction_freq': 'voter_item_freq'})
    train_samples = train_samples.merge(
        voter_item_freq_df,
        on=['voter_id', 'item_id'], how='left'
    )
else:
    print("警告: user_item_freq_orig 文件列名不符合预期，voter_item_freq 可能不正确。")
    train_samples['voter_item_freq'] = 0

# 合并类别信息到训练样本
if 'cate1_id' not in item_info.columns and 'cate_id' in item_info.columns:
    print("警告: item_info 中缺少 cate1_id，将使用 cate_id 作为 cate1_id 的代理。")
    item_info['cate1_id'] = item_info['cate_id']

if 'item_id' in item_info.columns and 'cate_id' in item_info.columns and 'cate1_id' in item_info.columns:
    item_info_to_merge = item_info[['item_id', 'cate_id', 'cate1_id']].copy()
    item_info_to_merge['cate_id'] = item_info_to_merge['cate_id'].astype(pd.StringDtype())
    item_info_to_merge['cate1_id'] = item_info_to_merge['cate1_id'].astype(pd.StringDtype())
    train_samples = train_samples.merge(item_info_to_merge, on='item_id', how='left')
else:
    print("警告: item_info 文件列名不符合预期 ('item_id', 'cate_id', 'cate1_id')，类别相关特征可能无法正确构建。")
    train_samples['cate_id'] = pd.NA
    train_samples['cate1_id'] = pd.NA

train_samples['cate_id'] = train_samples['cate_id'].astype(pd.StringDtype())

# inviter_cate_freq 和 voter_cate_freq
user_col_cate = None
cate_col_cate = None
freq_col_cate = None
if 'user_cate_freq_orig' in locals() and not user_cate_freq_orig.empty:
    actual_cols = user_cate_freq_orig.columns.tolist()
    if 'user_id' in actual_cols:
        user_col_cate = 'user_id'
    elif 'inviter_id' in actual_cols:
        user_col_cate = 'inviter_id'
    elif len(actual_cols) > 0:
        user_col_cate = actual_cols[0]

    if 'cate_id' in actual_cols:
        cate_col_cate = 'cate_id'
    elif len(actual_cols) > 1:
        cate_col_cate = actual_cols[1]

    if 'interaction_freq' in actual_cols:
        freq_col_cate = 'interaction_freq'
    elif 'count' in actual_cols:
        freq_col_cate = 'count'
    elif len(actual_cols) > 2:
        freq_col_cate = actual_cols[2]

if user_col_cate and cate_col_cate and freq_col_cate:
    user_cate_freq_orig_copy = user_cate_freq_orig.copy()
    user_cate_freq_orig_copy[cate_col_cate] = user_cate_freq_orig_copy[cate_col_cate].astype(pd.StringDtype())

    inviter_cate_df = user_cate_freq_orig_copy[[user_col_cate, cate_col_cate, freq_col_cate]].rename(columns={
        user_col_cate: 'inviter_id', cate_col_cate: 'cate_id', freq_col_cate: 'inviter_cate_freq'
    })
    train_samples = train_samples.merge(inviter_cate_df, on=['inviter_id', 'cate_id'], how='left')

    voter_cate_df = user_cate_freq_orig_copy[[user_col_cate, cate_col_cate, freq_col_cate]].rename(columns={
        user_col_cate: 'voter_id', cate_col_cate: 'cate_id', freq_col_cate: 'voter_cate_freq'
    })
    train_samples = train_samples.merge(voter_cate_df, on=['voter_id', 'cate_id'], how='left')
else:
    print(
        f"警告: user_cate_freq_orig 列名不符合预期或不完整。检测到的列: user_col='{user_col_cate}', cate_col='{cate_col_cate}', freq_col='{freq_col_cate}'。inviter_cate_freq 和 voter_cate_freq 将为0。")
    train_samples['inviter_cate_freq'] = 0
    train_samples['voter_cate_freq'] = 0

# 3. 计算比例和组合特征
print("计算比例和组合特征...")
cols_to_fill_for_ratio = [
    'voter_item_freq', 'inviter_item_freq',
    'voter_voted_count', 'inviter_voted_count',
    'voter_invite_count', 'inviter_invite_count',
    'voter_in_degree', 'inviter_in_degree',
    'voter_out_degree', 'inviter_out_degree',
    'user_interaction_freq', 'voter_cate_freq',  # 'voter_cate1_freq' 已移除
    'inviter_in_degree', 'inviter_out_degree'  # for inviter_influence
]
for col in cols_to_fill_for_ratio:
    if col not in train_samples.columns:
        train_samples[col] = 0
    else:
        train_samples[col].fillna(0, inplace=True)

# item_freq_ratio
train_samples['item_freq_ratio'] = np.where(
    (train_samples['inviter_item_freq'] > 0),
    train_samples['voter_item_freq'] / train_samples['inviter_item_freq'],
    0
)
# voted_count_ratio
train_samples['voted_count_ratio'] = np.where(
    (train_samples['inviter_voted_count'] > 0),
    train_samples['voter_voted_count'] / train_samples['inviter_voted_count'],
    0
)
# invite_count_ratio
train_samples['invite_count_ratio'] = np.where(
    (train_samples['inviter_invite_count'] > 0),
    train_samples['voter_invite_count'] / train_samples['inviter_invite_count'],
    0
)
# in_degree_ratio
train_samples['in_degree_ratio'] = np.where(
    (train_samples['inviter_in_degree'] > 0),
    train_samples['voter_in_degree'] / train_samples['inviter_in_degree'],
    0
)
# out_degree_ratio
train_samples['out_degree_ratio'] = np.where(
    (train_samples['inviter_out_degree'] > 0),
    train_samples['voter_out_degree'] / train_samples['inviter_out_degree'],
    0
)

# interaction_strength
train_samples['interaction_strength'] = (
        train_samples['user_interaction_freq'] * 5 +
        train_samples['voter_item_freq'] * 2 +
        train_samples['voter_cate_freq'] * 1  # + train_samples['voter_cate1_freq'] * 0.5 已移除
)

# social_influence (voter)
train_samples['social_influence'] = train_samples['voter_in_degree'] * train_samples['voter_out_degree']
# inviter_influence
train_samples['inviter_influence'] = train_samples['inviter_in_degree'] * train_samples['inviter_out_degree']

# 最终填充所有特征的NaN为0
final_feature_cols_definition = [
    'user_interaction_freq',
    'interaction_strength',
    'voter_voted_count',
    'inviter_out_degree',
    'inviter_invite_count',
    'voter_cate_freq',
    'item_popularity',
    'inviter_cate_freq',
    'inviter_item_freq',
    'inviter_voted_count',
    'voted_count_ratio',
    'voter_invite_count',
    'in_degree_ratio',
    'invite_count_ratio',
    'inviter_influence'
]
for col in final_feature_cols_definition:
    if col not in train_samples.columns:
        print(f"警告: 特征列 '{col}' 在训练样本中缺失，将创建并填充0。")
        train_samples[col] = 0
    else:
        train_samples[col].fillna(0, inplace=True)

# 提取最终的特征列和标签
feature_cols = final_feature_cols_definition
X = train_samples[feature_cols]
y = train_samples['label']

# 检查特征中是否有NaN或无穷大值
if X.isnull().values.any():
    print("警告: 特征数据 X 中包含 NaN 值。请检查特征工程步骤。")
    X = X.fillna(0)  # 再次填充以防万一
if np.isinf(X.values).any():
    print("警告: 特征数据 X 中包含无穷大值。请检查特征工程步骤。")
    X = X.replace([np.inf, -np.inf], 0)  # 替换无穷大值

# 划分训练集和验证集
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 保持标签比例
    )
    log_message(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
except Exception as e:
    log_message(f"划分训练集和验证集时发生错误: {e}")
    sys.exit(1)

log_message("开始训练 LightGBM 模型...")

# 1. 网格搜索参数范围（适当缩小）
param_grid = {
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [2, 3, 4],
    'num_leaves': [26, 31, 33],
    'subsample': [0.7, 0.8, 0.85],
    'colsample_bytree': [0.75, 0.8, 0.9],
    'n_estimators': [80, 100, 110]
}

lgb_estimator = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42,
    n_jobs=1,  # 防止内存溢出
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1
)

try:
    log_message("开始参数网格搜索...")
    grid = GridSearchCV(
        estimator=lgb_estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,  # 3折交叉验证
        verbose=2,
        n_jobs=1
    )
    grid.fit(X_train, y_train)
    log_message(f"最优参数: {grid.best_params_}")
    log_message(f"最优AUC: {grid.best_score_}")
except Exception as e:
    log_message(f"参数网格搜索时发生错误: {e}")
    sys.exit(1)

# 2. 用最优参数+早停机制训练最终模型
try:
    log_message("使用最优参数重新训练最终模型（带早停）...")
    best_params = grid.best_params_
    model = lgb.LGBMClassifier(
        **best_params,
        objective='binary',
        metric='auc',
        random_state=42,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1
    )
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    end_time = time.time()
    training_time = end_time - start_time
    log_message(f"最终模型训练完成，耗时: {training_time:.2f}秒")
except Exception as e:
    log_message(f"最终模型训练时发生错误: {e}")
    sys.exit(1)

# 评估模型
try:
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    log_message(f"验证集 AUC: {auc_score:.5f}")
except Exception as e:
    log_message(f"模型评估时发生错误: {e}")
    sys.exit(1)

# 显示特征重要性
try:
    feature_importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    log_message("\n特征重要性:")
    log_message(feature_importance_df.head(30).to_string())  # 显示所有特征
    feature_importance_df.to_csv('results/features/training_feature_importance.csv', index=False)
except Exception as e:
    log_message(f"生成特征重要性时发生错误: {e}")

# 保存模型
model_path = 'results/models/lgb_model.pkl'
log_message(f"保存模型到 {model_path} ...")
try:
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log_message("模型保存成功！")
except Exception as e:
    log_message(f"保存模型时发生错误: {e}")
    sys.exit(1)

log_message("模型训练完成！")