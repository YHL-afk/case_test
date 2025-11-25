import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0. 读数据 & 性别数字化
# =========================

file_path = r"E:\project1\CaseStudy_health_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

# 性别数字化
gender_map = {
    "Male": 1, "Female": 0,
    "male": 1, "female": 0,
    "M": 1, "F": 0,
}
df["gender_num"] = df["gender"].map(gender_map)

# 因变量
y_col = "disease_risk"
if y_col not in df.columns:
    raise ValueError("缺少 disease_risk 列")

# =========================
# 1. 手动区分连续 / 二元变量
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

# 检查列是否存在
for c in cont_cols + bin_cols + [y_col]:
    if c not in df.columns:
        raise ValueError(f"列 {c} 在数据中不存在: {c}")

print("\n连续变量：", cont_cols)
print("二元变量：", bin_cols)

# =========================
# 2. 二元变量：统计 X=1 / X=0 的 risk 概率
# =========================

binary_results = []

for col in bin_cols:
    sub1 = df[df[col] == 1]   # X=1 视为“高侧”
    sub0 = df[df[col] == 0]   # X=0 视为“低侧”

    if len(sub1) == 0 or len(sub0) == 0:
        binary_results.append({
            "feature": col,
            "type": "binary",
            "risk_high": np.nan,   # X=1 的风险概率
            "risk_low": np.nan,    # X=0 的风险概率
            "n_high": len(sub1),
            "n_low": len(sub0),
            "worst_side": None,
            "worst_risk": np.nan,
        })
        continue

    p1 = sub1[y_col].mean()  # P(risk=1 | X=1)
    p0 = sub0[y_col].mean()  # P(risk=1 | X=0)

    # 最危险一侧及其概率（只看大小，不看方向）
    if p1 >= p0:
        worst_side = "high"      # 高侧 / X=1 更危险
        worst_risk = p1
    else:
        worst_side = "low"       # 低侧 / X=0 更危险
        worst_risk = p0

    binary_results.append({
        "feature": col,
        "type": "binary",
        "risk_high": p1,
        "risk_low": p0,
        "n_high": len(sub1),
        "n_low": len(sub0),
        "worst_side": worst_side,
        "worst_risk": worst_risk,
    })

bin_df = pd.DataFrame(binary_results)
print("\n二元变量单变量风险统计：")
print(bin_df)

# =========================
# 3. 连续变量：按均值 ± 1σ 取两端的 risk 概率
#    高侧: x >= m + s
#    低侧: x <= m - s
# =========================

cont_results = []

for col in cont_cols:
    x = df[col].values
    y = df[y_col].values

    m = np.mean(x)
    s = np.std(x)  # 标准差 σ

    high_group = df[x >= m + s]   # 明显偏高
    low_group  = df[x <= m - s]   # 明显偏低

    if len(low_group) == 0 or len(high_group) == 0:
        cont_results.append({
            "feature": col,
            "type": "continuous",
            "risk_high": np.nan,
            "risk_low": np.nan,
            "n_high": len(high_group),
            "n_low": len(low_group),
            "mean": m,
            "std": s,
            "worst_side": None,
            "worst_risk": np.nan,
        })
        continue

    risk_high = high_group[y_col].mean()  # P(risk=1 | x >= m+s)
    risk_low = low_group[y_col].mean()    # P(risk=1 | x <= m-s)

    if risk_high >= risk_low:
        worst_side = "high"   # 值偏高那边更危险
        worst_risk = risk_high
    else:
        worst_side = "low"    # 值偏低那边更危险
        worst_risk = risk_low

    cont_results.append({
        "feature": col,
        "type": "continuous",
        "risk_high": risk_high,
        "risk_low": risk_low,
        "n_high": len(high_group),
        "n_low": len(low_group),
        "mean": m,
        "std": s,
        "worst_side": worst_side,
        "worst_risk": worst_risk,
    })

cont_df = pd.DataFrame(cont_results)
print("\n连续变量单变量风险统计（>=均值+1σ vs <=均值-1σ）：")
print(cont_df)

# =========================
# 4. 合并 + 构造有符号的“风险条形值”
# =========================

all_df = pd.concat([cont_df, bin_df], ignore_index=True)

# 有符号的条形值：高侧更危险 → +worst_risk；低侧更危险 → -worst_risk
def signed_risk(row):
    wr = row["worst_risk"]
    if pd.isna(wr):
        return 0.0
    if row["worst_side"] == "high":
        return +wr
    elif row["worst_side"] == "low":
        return -wr
    else:
        return 0.0

all_df["signed_risk"] = all_df.apply(signed_risk, axis=1)

# 绝对风险，用于排序和折线图
all_df["abs_worst_risk"] = all_df["worst_risk"].abs()

# 按“最坏侧风险概率绝对值”从低到高排序
all_sorted = all_df.sort_values("abs_worst_risk")

print("\n=== 按最坏侧 risk 概率从低到高排序 ===")
print(all_sorted[[
    "feature", "type",
    "risk_high", "risk_low",
    "worst_side", "worst_risk",
    "signed_risk"
]])

# =========================
# 5. 图 1：有方向的概率条形图（左：低值更危险，右：高值/取1更危险）
# =========================

plt.figure(figsize=(9, 6))

signed_vals = all_sorted["signed_risk"].values
colors = [
    "tab:red" if v > 0 else
    "tab:blue" if v < 0 else
    "grey"
    for v in signed_vals
]

plt.barh(all_sorted["feature"], signed_vals, color=colors)
plt.axvline(0, color="black", linewidth=1)

# x 轴以概率为单位，范围自动，也可以手动限制 [−1,1]
plt.xlabel("Signed worst-side P(disease_risk=1)")
plt.title(
    "Directional worst-side risk by feature\n"
)
plt.tight_layout()
plt.show()

# =========================
# 6. 图 2：折线图 —— 按最坏侧风险概率大小排序
# =========================

features = all_sorted["feature"].tolist()
worst_risks = all_sorted["abs_worst_risk"].values  # 这里用绝对值，其实=worst_risk本身(非负)

x_idx = np.arange(len(features))

plt.figure(figsize=(10, 6))
plt.plot(x_idx, worst_risks, marker="o")

plt.xticks(x_idx, features, rotation=45, ha="right")
plt.ylabel("P(disease_risk=1) on worst side")
plt.xlabel("Features (sorted by worst-side risk)")
plt.title("Worst-side disease risk by feature (sorted by magnitude)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
