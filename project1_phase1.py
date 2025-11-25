import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# ========= 1. 读取数据 =========
file_path = r"E:\project1\CaseStudy_health_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

print("原始列名:", df.columns.tolist())
print(df.head())

# ========= 2. 性别数字化：Male -> 1, Female -> 0 =========
gender_map = {
    "Male": 1,
    "Female": 0,
    "male": 1,
    "female": 0,
    "M": 1,
    "F": 0,
}

df["gender_num"] = df["gender"].map(gender_map)

# 如果有没映射上的值，可以检查一下
if df["gender_num"].isna().any():
    print("⚠️ 性别列中有无法映射的值，请检查：")
    print(df[["gender", "gender_num"]][df["gender_num"].isna()].drop_duplicates())

print("\n性别数字化后的前几行：")
print(df[["gender", "gender_num"]].head())

# ========= 3. 选择特征列：除了因变量 disease_risk，其他都作为自变量 =========
# 一般不建议把 id 这种纯标识符算进去，所以排除 id、原始 gender 字符列、本身的 risk 标签
cols_to_exclude = ["id", "gender", "disease_risk"]

# 如果你还有别的明确不想进模型的列，也可以加进来：
# cols_to_exclude += ["某些不想用的列名"]

feature_cols = [c for c in df.columns if c not in cols_to_exclude]

print("\n用于 GMM 的特征列：", feature_cols)

# 检查一下 disease_risk 是否存在
if "disease_risk" not in df.columns:
    raise ValueError("数据中没有 'disease_risk' 这一列，请检查原始文件。")

# ========= 4. 标准化所有自变量（z-score） =========
scaler = StandardScaler()

X_features = df[feature_cols]
X_scaled = scaler.fit_transform(X_features)

df_scaled = df.copy()
df_scaled[feature_cols] = X_scaled

print("\n标准化后的前几行（特征 + disease_risk）：")
print(df_scaled[feature_cols + ["disease_risk"]].head())

# 后续做 GMM 时用的特征矩阵
X_for_gmm = df_scaled[feature_cols].values

# ========= 5. 划分训练集 / 验证集（80% / 20%） =========
X_train, X_val = train_test_split(
    X_for_gmm,
    test_size=0.2,
    random_state=0,
    shuffle=True
)

print("\n训练集大小:", X_train.shape)
print("验证集大小:", X_val.shape)

# ========= 6. 多 K 的 GMM + 训练集 BIC + 验证集 log-likelihood =========
K_range = range(1, 12)  # K = 1 ~ 12
bic_train = []
val_loglik = []  # 验证集上的平均 log-likelihood（越大越好）

for K in K_range:
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=0,
        n_init=10
    )
    gmm.fit(X_train)

    # 训练集上的 BIC
    bic_k = gmm.bic(X_train)
    bic_train.append(bic_k)

    # 验证集上的平均 log-likelihood
    val_ll_k = gmm.score(X_val)  # 平均 log p(x_i)
    val_loglik.append(val_ll_k)

print("\n候选 K:", list(K_range))
print("训练集 BIC:", bic_train)
print("验证集 平均 log-likelihood:", val_loglik)

# 按 BIC 找最小的那个 K
bic_best_index = int(np.argmin(bic_train))
best_K_bic = list(K_range)[bic_best_index]

print(f"\n>>> 按 BIC 最优的 K = {best_K_bic}")

# ========= 7. 画折线图：BIC / 验证集 log-likelihood =========
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(list(K_range), bic_train, marker="o")
plt.xlabel("K")
plt.ylabel("BIC (train)")
plt.title("BIC vs K")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(list(K_range), val_loglik, marker="o")
plt.xlabel("K")
plt.ylabel("Avg log-likelihood (val)")
plt.title("Validation log-likelihood vs K")
plt.grid(True)

plt.tight_layout()
plt.show()

# ========= 8. 选择最终 K，并在全部数据上拟合最终 GMM =========
final_K = best_K_bic
print(f"\n>>> 选择最终簇数 K = {final_K}（按训练集 BIC）")

final_gmm = GaussianMixture(
    n_components=final_K,
    covariance_type="full",
    random_state=0,
    n_init=20
)
final_gmm.fit(X_for_gmm)  # 在全体数据上拟合最终模型

# 每个样本的簇标签（硬划分）
cluster_labels = final_gmm.predict(X_for_gmm)
df_scaled["cluster"] = cluster_labels

print("\n簇标签前几行：")
if "id" in df_scaled.columns:
    print(df_scaled[["id", "cluster", "disease_risk"]].head())
else:
    print(df_scaled[["cluster", "disease_risk"]].head())

# 如需保存结果：
# out_path = r"E:\project1\CaseStudy_health_lifestyle_allvars_clusters.csv"
# df_scaled.to_csv(out_path, index=False)
# print(f"\n已保存带簇标签的数据到: {out_path}")
