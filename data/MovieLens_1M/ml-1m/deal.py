# import pandas as pd
# import numpy as np
# import os

# # -------------------------------
# # 配置文件路径
# # -------------------------------
# import pandas as pd

# data_dir = "/Users/mia/Documents/machine learning/final/ReChorus/data/MovieLens_1M/ml-1m"

# df_train = pd.read_csv(f"{data_dir}/train.csv", sep='\t', engine='python', skiprows=1,
#                        names=['user_id','item_id','rating','time'], dtype=int)
# df_dev   = pd.read_csv(f"{data_dir}/dev.csv", sep='\t', engine='python', skiprows=1,
#                        names=['user_id','item_id','rating','time'], dtype=int)
# df_test  = pd.read_csv(f"{data_dir}/test.csv", sep='\t', engine='python', skiprows=1,
#                        names=['user_id','item_id','rating','time'], dtype=int)

# print(df_train.shape, df_dev.shape, df_test.shape)
# n_users = max(df_train['user_id'].max(), df_dev['user_id'].max(), df_test['user_id'].max()) + 1
# n_items = max(df_train['item_id'].max(), df_dev['item_id'].max(), df_test['item_id'].max()) + 1

# print(n_users, n_items)

# num_neg = 99  # 每个样本负样本数量

# # 总用户数和物品数
# print(f"#users: {n_users}, #items: {n_items}")

# # -------------------------------
# # 构建每个用户的正样本集合
# # -------------------------------
# user_pos_dict = df_train.groupby('user_id')['item_id'].apply(set).to_dict()

# # -------------------------------
# # 生成负样本函数
# # -------------------------------
# def generate_neg_items(df_eval, n_items, user_pos_dict, num_neg=99):
#     neg_items_list = []
#     for _, row in df_eval.iterrows():
#         user = row['user_id']
#         pos_items = user_pos_dict.get(user, set())
#         neg_candidates = list(set(range(n_items)) - pos_items)
#         if len(neg_candidates) >= num_neg:
#             sampled = np.random.choice(neg_candidates, size=num_neg, replace=False)
#         else:
#             sampled = np.random.choice(neg_candidates, size=num_neg, replace=True)
#         neg_items_list.append(sampled)
#     df_eval['neg_items'] = neg_items_list
#     return df_eval

# # -------------------------------
# # 生成 dev/test 的 neg_items
# # -------------------------------
# df_dev = generate_neg_items(df_dev, n_items, user_pos_dict, num_neg)
# df_test = generate_neg_items(df_test, n_items, user_pos_dict, num_neg)

# # -------------------------------
# # 保存回 CSV
# # -------------------------------
# df_dev.to_csv(os.path.join(data_dir, "dev_neg.csv"), index=False)
# df_test.to_csv(os.path.join(data_dir, "test_neg.csv"), index=False)

# print("neg_items 已生成并保存完毕")
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

# -----------------------------
# 配置路径
# -----------------------------
data_dir = "/Users/mia/Documents/machine learning/final/ReChorus/data/MovieLens_1M/ml-1m"
output_dir = data_dir                # 输出目录，可改

# 读取原始数据
df_train = pd.read_csv(f"{data_dir}/train.csv", sep='\t', engine='python', skiprows=1,
                       names=['user_id','item_id','rating','time'], dtype=int)
df_dev   = pd.read_csv(f"{data_dir}/dev.csv", sep='\t', engine='python', skiprows=1,
                       names=['user_id','item_id','rating','time'], dtype=int)
df_test  = pd.read_csv(f"{data_dir}/test.csv", sep='\t', engine='python', skiprows=1,
                       names=['user_id','item_id','rating','time'], dtype=int)

# 用户和物品总数
n_users = max(df_train['user_id'].max(), df_dev['user_id'].max(), df_test['user_id'].max()) + 1
n_items = max(df_train['item_id'].max(), df_dev['item_id'].max(), df_test['item_id'].max()) + 1

# 为每个用户生成训练集正样本集合
user_train_dict = df_train.groupby('user_id')['item_id'].apply(set).to_dict()

# -----------------------------
# 生成负采样列表
# -----------------------------
def generate_neg_items(user_id, n_neg=100):
    pos_items = user_train_dict.get(user_id, set())
    neg_items = []
    while len(neg_items) < n_neg:
        neg = random.randint(0, n_items-1)
        if neg not in pos_items:
            neg_items.append(neg)
    return neg_items

def add_neg_items(df):
    neg_list = []
    for uid in tqdm(df['user_id'].tolist()):
        neg_list.append(generate_neg_items(uid, n_neg=100))
    df['neg_items'] = neg_list
    return df

# -----------------------------
# 处理 train/dev/test
# -----------------------------
df_train_new = add_neg_items(df_train)
df_dev_new = add_neg_items(df_dev)
df_test_new = add_neg_items(df_test)

# -----------------------------
# 保存为 CSV，确保每行完整
# -----------------------------
def save_df(df, name):
    out_path = os.path.join(output_dir, f"{name}_with_neg.tsv")
    df.to_csv(out_path, sep='\t', index=False)
    print(f"{name} saved to {out_path}")
    
save_df(df_dev_new, "dev")
save_df(df_test_new, "test")
