import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D  # 触发 3D 绘图

# =========================
# 0. 基本配置
# =========================

RAW_CSV_PATH = r"E:\project1\CaseStudy_health_lifestyle_dataset.csv"

SCALED_SAVE_PATH = r"E:\project1\CaseStudy_health_lifestyle_scaled_k11.csv"
CLUSTER_SAVE_PATH = r"E:\project1\CaseStudy_health_lifestyle_k11_clusters.csv"

K = 10  # 固定 10 类；想改回 7 等，改这一行即可

# =========================
# 1. 读数据 + 性别数字化
# =========================

df = pd.read_csv(RAW_CSV_PATH)
print("原始列名：", df.columns.tolist())
print(df.head())

gender_map = {
    "Male": 1, "Female": 0,
    "male": 1, "female": 0,
    "M": 1, "F": 0,
}
df["gender_num"] = df["gender"].map(gender_map)

if df["gender_num"].isna().any():
    print("\n⚠️ 性别有无法映射的取值，请检查：")
    print(df[["gender", "gender_num"]][df["gender_num"].isna()].drop_duplicates())

# =========================
# 2. 手动区分连续特征 / 二元特征
# =========================

cont_cols = [
    "age",
    "daily_steps",
    "sleep_hours",
    "water_intake_l",
    "calories_consumed",
    "bmi",
    "resting_hr",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
]

bin_cols = [
    "smoker",
    "alcohol",
    "family_history",
    "gender_num",
]

overlay_col = "disease_risk"  # 只用于风险分布，不参与聚类

for c in cont_cols + bin_cols + [overlay_col]:
    if c not in df.columns:
        raise ValueError(f"列 {c} 在数据里不存在，请检查: {c}")

print("\n连续特征：", cont_cols)
print("二元特征：", bin_cols)

# =========================
# 3. 聚类用的特征：连续做 z-score，二元保持 0/1
# =========================

scaler = StandardScaler()
X_cont_scaled = scaler.fit_transform(df[cont_cols])   # 连续 → 标准化
X_bin = df[bin_cols].values.astype(float)            # 二元 → 原样 0/1

X_all = np.hstack([X_cont_scaled, X_bin])

# 存一份“标准化后 + 0/1”的数据，仅作记录
df_scaled = df.copy()
df_scaled[cont_cols] = X_cont_scaled
df_scaled.to_csv(SCALED_SAVE_PATH, index=False)
print(f"\n已保存标准化数据到: {SCALED_SAVE_PATH}")

# =========================
# 4. 拟合 GMM(K)，给所有样本打簇标签
# =========================

gmm_final = GaussianMixture(
    n_components=K,
    covariance_type="full",
    random_state=0,
    n_init=20,
)
gmm_final.fit(X_all)

clusters = gmm_final.predict(X_all)  # 0 ~ K-1
df["cluster"] = clusters
df_scaled["cluster"] = clusters

bic_all = gmm_final.bic(X_all)
avg_ll_all = gmm_final.score(X_all)
print(f"\nK={K} 模型: BIC = {bic_all:.2f}, Avg log-likelihood = {avg_ll_all:.3f}")

print("\n带簇标签的前几行：")
if "id" in df.columns:
    print(df[["id", "cluster", overlay_col]].head())
else:
    print(df[["cluster", overlay_col]].head())

df.to_csv(CLUSTER_SAVE_PATH, index=False)
print(f"\n已保存带簇标签的原始数值数据到: {CLUSTER_SAVE_PATH}")

# =========================
# 5. 图一：Cluster × 特征热力图（连续 z-score + 二元映射到 -1/1）
# =========================

feature_cols_all = cont_cols + bin_cols

# 仅用于“画图”的矩阵：
# 连续列用 z-score，二元列 0→-1, 1→+1
heat_df = pd.DataFrame(index=df.index)
heat_df[cont_cols] = X_cont_scaled                 # 已标准化的连续特征
heat_df[bin_cols] = df[bin_cols] * 2.0 - 1.0       # 0/1 → -1/+1
heat_df["cluster"] = clusters

cluster_heat_means = (
    heat_df
    .groupby("cluster")[feature_cols_all]
    .mean()
    .sort_index()
)

print("\n用于热力图的簇均值（连续=z-score，二元≈[-1,1]）：")
print(cluster_heat_means)

# 热力图
fig1, ax1 = plt.subplots(figsize=(12, 6))

im = ax1.imshow(
    cluster_heat_means[feature_cols_all].values,
    aspect="auto",
    vmin=-2.0, vmax=2.0
)
cbar = fig1.colorbar(im, ax=ax1,
                     label="Level (continuous: z-score, binary: -1~1)")

ax1.set_yticks(np.arange(cluster_heat_means.shape[0]))
ax1.set_yticklabels([f"Cluster {c}" for c in cluster_heat_means.index])

ax1.set_xticks(np.arange(len(feature_cols_all)))
ax1.set_xticklabels(feature_cols_all, rotation=45, ha="right")

ax1.set_title("Cluster × feature pattern")
fig1.tight_layout()

# =========================
# 5b. 3D 图：每个簇在各特征上的均值（你之前喜欢的那个版本）
# =========================

# =========================
# 5b. 3D 图：每个簇在各特征上的均值
# =========================

fig_3d = plt.figure(figsize=(10, 7))
ax3d = fig_3d.add_subplot(111, projection="3d")

n_clusters = cluster_heat_means.shape[0]
n_features = len(feature_cols_all)

for _, cluster_id in enumerate(cluster_heat_means.index):
    xs = np.arange(n_features)  # 特征索引
    ys = np.full_like(xs, cluster_id, dtype=float)  # 当前 cluster 的一条水平线
    zs = cluster_heat_means.loc[cluster_id, feature_cols_all].values

    ax3d.scatter(xs, ys, zs, s=40, label=f"Cluster {cluster_id}")

# 横坐标只保留刻度和特征名，不要“Feature”文字
ax3d.set_xlabel("")   # 不要写 "Feature"
ax3d.set_ylabel("Cluster")
ax3d.set_zlabel("Cluster mean\n(z-score / -1~1)")

ax3d.set_xticks(np.arange(n_features))
ax3d.set_xticklabels(feature_cols_all, rotation=45, ha="right")

ax3d.set_yticks(cluster_heat_means.index)
ax3d.set_yticklabels(cluster_heat_means.index)

ax3d.set_title("3D view of cluster-wise feature means")

# 图例放在右下角
ax3d.legend(loc="lower right", bbox_to_anchor=(1.25, 0.05))

fig_3d.tight_layout()


# =========================
# 6. 图二：各簇 disease_risk 分布
#    左：折线图（点上标数值，图例右下角）
#    右：饼图（人数占比 + 右侧图例）
# =========================

risk_stats_full = (
    df
    .groupby("cluster")[overlay_col]
    .agg(["mean", "count", "sum"])   # mean=P(risk=1), sum=高风险数量
    .rename(columns={"mean": "risk_rate",
                     "count": "n",
                     "sum": "n_risk"})
    .sort_index()
)

print("\n各簇 disease_risk 统计：")
print(risk_stats_full)

# 全体样本的总体 risk 期望（全局均值）
overall_risk_mean = df[overlay_col].mean()
print(f"\n总体 risk 期望（全体样本 P(disease_risk=1)）：{overall_risk_mean:.4f}")

# 用 constrained_layout 避免饼图和图例被挤掉
fig2, axes = plt.subplots(
    1, 2, figsize=(13, 5), constrained_layout=True
)

# ------ 左：折线图 P(risk=1) + 总体期望线 ------

x_idx = risk_stats_full.index.values
y_risk = risk_stats_full["risk_rate"].values

axes[0].plot(
    x_idx,
    y_risk,
    marker="o",
    label="Cluster-wise P(risk=1)"
)

# 文字偏移量，避免被 y 轴裁掉
y_min, y_max = y_risk.min(), y_risk.max()
offset = (y_max - y_min) * 0.03 if y_max > y_min else 0.001

for x, y in zip(x_idx, y_risk):
    axes[0].text(
        x, y + offset,
        f"{y:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

axes[0].set_ylim(y_min - 2 * offset, y_max + 4 * offset)

# 总体期望：水平虚线
axes[0].axhline(
    overall_risk_mean,
    color="red",
    linestyle="--",
    label=f"Overall P(risk=1) = {overall_risk_mean:.3f}"
)

axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("P(disease_risk = 1)")
axes[0].set_title("Disease risk rate by cluster")
axes[0].grid(True, alpha=0.3)

# 图例放在右下角
axes[0].legend(loc="lower right")

# ------ 右：饼图（每簇人数占比，比例写在右侧图例里） ------

axes[1].set_title("Population share by cluster")

cluster_sizes = risk_stats_full["n"].values          # 每簇人数
labels = [f"C{c}" for c in risk_stats_full.index]    # C0, C1, ...

if cluster_sizes.sum() > 0:
    wedges, _ = axes[1].pie(
        cluster_sizes,
        startangle=90
    )
    axes[1].set_aspect("equal")

    total_n = cluster_sizes.sum()
    legend_labels = [
        f"{lab} ({n / total_n * 100:.1f}%)"
        for lab, n in zip(labels, cluster_sizes)
    ]

    axes[1].legend(
        wedges,
        legend_labels,
        title="Cluster (population share)",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5)   # 右侧，不挡图
    )
else:
    axes[1].text(0.5, 0.5, "No data",
                 ha="center", va="center")
    axes[1].set_axis_off()

# =========================
# 7. 图三：还原到标准化之前的簇均值表（+ 每簇人数 + risk 概率）
# =========================

# 还原连续变量：簇级 z-score → 原始单位
cont_means_z = cluster_heat_means[cont_cols].values
cont_means_raw = scaler.inverse_transform(cont_means_z)
cont_means_raw_df = pd.DataFrame(
    cont_means_raw,
    index=cluster_heat_means.index,
    columns=cont_cols
)

# 二元变量：直接用原始 df 算每簇 1 的比例
bin_means_prop = (
    df
    .groupby("cluster")[bin_cols]
    .mean()
    .sort_index()
)

# 每簇人数 & risk 概率
cluster_sizes_series = risk_stats_full["n"].rename("n_samples")
cluster_risk_rate = risk_stats_full["risk_rate"].rename("risk_rate")

# 拼成一个总表：人数 + risk 概率 + 连续原始均值 + 二元比例
cluster_means_raw = pd.concat(
    [cluster_sizes_series, cluster_risk_rate, cont_means_raw_df, bin_means_prop],
    axis=1
)

print("\n还原到原始单位的簇均值表（含每簇人数 + risk 概率）：")
print(cluster_means_raw)

# 可视化为表格
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.axis("off")

tbl = ax3.table(
    cellText=np.round(cluster_means_raw.values, 3),
    rowLabels=[f"Cluster {c}" for c in cluster_means_raw.index],
    colLabels=cluster_means_raw.columns,
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.4)

ax3.set_title(
    "Cluster-level stats (n + risk_rate + continuous in original units, binary as proportion)",
    pad=10
)

fig3.tight_layout()

# =========================
# 一次性展示所有图
# =========================

plt.show()
