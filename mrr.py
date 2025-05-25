import pandas as pd
from joblib import Parallel, delayed
import json
import numpy as np
from tqdm import tqdm


def calculate_metrics_per_batch(i, predictions_batch, test_data_dict):
    results = []
    for prediction in tqdm(predictions_batch, desc=f"batch {i}", position=1, leave=True):
        triple_id = prediction["triple_id"]
        candidate_voter_list = prediction["candidate_voter_list"]

        # 用字典查找真实voter，效率高
        if triple_id in test_data_dict:
            x = test_data_dict[triple_id]
            true_voter = str(x['voter_id'])
            inviter_id = x['inviter_id']
            item_id = x['item_id']
            timestamp = x['timestamp']
        else:
            true_voter = inviter_id = item_id = timestamp = None

        if true_voter in candidate_voter_list:
            rank = candidate_voter_list.index(true_voter) + 1
            mrr = 1.0 / rank
            hit_at_5 = 1
        else:
            mrr = 0.0
            hit_at_5 = 0

        results.append({
            "triple_id": triple_id,
            'inviter_id': inviter_id,
            'item_id': item_id,
            'timestamp': timestamp,
            "predicted_voters": candidate_voter_list,
            "predicted_len": len(candidate_voter_list),
            "true_voter": true_voter,
            "mrr": mrr,
            "hits_at_5": hit_at_5
        })
    return results


def calculate_metrics(test_data, predictions):
    # 构建triple_id到样本的字典
    test_data_dict = {str(x['triple_id']): x for x in test_data}
    # 分批
    predictions_batches = np.array_split(predictions, 12)
    # 并行处理
    results_batches = Parallel(n_jobs=12)(
        delayed(calculate_metrics_per_batch)(i, predictions_batches[i], test_data_dict) for i in range(len(predictions_batches)))
    # 合并
    results = [item for sublist in results_batches for item in sublist]
    # 计算总的MRR和HITS@5
    total_mrr = sum(result['mrr'] for result in results) / len(predictions)
    total_hits_at_5 = sum(result['hits_at_5'] for result in results) / len(predictions)
    return total_mrr, total_hits_at_5, results


# 读取训练集和预测结果  
with open('data/item_share_train_info_B.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
# 添加triple_id字段
for idx, x in enumerate(test_data):
    x['triple_id'] = str(idx)

with open('results/output/predictions_parallel.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

# 计算指标
total_mrr, total_hits_at_5, results = calculate_metrics(test_data, predictions)

print(f"总的MRR: {total_mrr:.5f}")
print(f"总的HITS@5: {total_hits_at_5:.5f}")
with open('results/output/evaluation_results.txt', 'w') as f:
    f.write(f"MRR@5: {total_mrr:.5f}\n")
    f.write(f"HITS@5: {total_hits_at_5:.5f}\n")

# 保存每一个query的结果到excel文件中
df = pd.DataFrame(results)
df.to_excel("./results.xlsx", index=False)

