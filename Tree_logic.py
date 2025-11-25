
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

CSV_PATH = r"E:\project1\CaseStudy_health_lifestyle_dataset.csv"


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)


    gender_map = {
        "Male": 1, "Female": 0,
        "male": 1, "female": 0,
        "M": 1, "F": 0,
    }
    if "gender" in df.columns:
        df["gender_num"] = df["gender"].map(gender_map)


    for col in df.columns:
        if df[col].dtype.kind in "biufc":  # 数值列
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df




def train_decision_tree(X_train, y_train, X_valid, y_valid):
    clf = DecisionTreeClassifier(
        max_depth=4,          
        min_samples_leaf=50,  
        class_weight='balanced',  
        random_state=42,
    )
    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_prob)
    print(f"[DecisionTree] validation AUC = {valid_auc:.4f}")

    return clf




def extract_tree_rules(tree: DecisionTreeClassifier, feature_names):
    """
    从 sklearn 决策树中提取每个叶节点对应的规则列表。
    返回: dict[leaf_node_id] = [rule1, rule2, ...]
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    rules = {}

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]


            left_cond = conditions + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_cond)

            right_cond = conditions + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_cond)
        else:

            rules[node] = conditions

    recurse(0, [])
    return rules




def build_segment_table(df, feature_cols, tree,
                        target_col="disease_risk"):

    X = df[feature_cols]
    leaf_ids = tree.apply(X)
    df = df.copy()
    df["segment_id"] = leaf_ids


    seg_stats = (
        df.groupby("segment_id")
        .agg(
            n_samples=(target_col, "size"),
            mean_true_risk=(target_col, "mean"),
        )
        .reset_index()
    )

    rules_dict = extract_tree_rules(tree, feature_cols)
    rules_rows = []
    for leaf_id, cond_list in rules_dict.items():
        rule_str = " AND ".join(cond_list) if cond_list else "(all samples)"
        rules_rows.append({"segment_id": leaf_id, "rules": rule_str})
    rules_df = pd.DataFrame(rules_rows)

    seg_table = seg_stats.merge(rules_df, on="segment_id", how="left")


    seg_table = seg_table.sort_values("mean_true_risk", ascending=False)

    return seg_table, df




def train_leaf_logistic_models(tree: DecisionTreeClassifier,
                               X_train: pd.DataFrame,
                               y_train: pd.Series):

    leaf_ids = tree.apply(X_train)
    unique_leaf_ids = np.unique(leaf_ids)

    leaf_models = {}

    for leaf_id in unique_leaf_ids:
        mask = (leaf_ids == leaf_id)
        X_leaf = X_train.loc[mask]
        y_leaf = y_train.loc[mask]


        if y_leaf.nunique() < 2:
            leaf_models[leaf_id] = None
            continue

        logit = LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )
        logit.fit(X_leaf, y_leaf)
        leaf_models[leaf_id] = logit

    return leaf_models




def main():

    df = load_and_preprocess(CSV_PATH)

    target_col = "disease_risk"


    drop_cols = [target_col]
    for c in ["gender", "id"]:
        if c in df.columns:
            drop_cols.append(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]


    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tree_clf = train_decision_tree(X_train, y_train, X_valid, y_valid)


    seg_table, df_with_segments = build_segment_table(
        df, feature_cols, tree_clf, target_col=target_col
    )

    print("\n===== 决策树 segment 列表（按真实平均风险从高到低）=====")
    print(seg_table.to_string(index=False))


    seg_table.to_csv("segments_summary_dt.csv", index=False)
    df_with_segments.to_csv("dataset_with_segments_dt.csv", index=False)


    seg_sorted = seg_table.sort_values("mean_true_risk", ascending=False)

    top_n = min(15, len(seg_sorted))
    seg_top = seg_sorted.iloc[:top_n].copy()
    seg_top["rank"] = np.arange(1, top_n + 1)

    overall_mean = df[target_col].mean()

    fig_line, ax_line = plt.subplots(figsize=(10, 4))
    ax_line.plot(
        seg_top["rank"],
        seg_top["mean_true_risk"],
        marker="o",
        label="Segment-wise P(risk=1)",
    )

    ax_line.axhline(
        overall_mean,
        color="red",
        linestyle="--",
        label=f"Overall mean = {overall_mean:.3f}",
    )

    for _, row in seg_top.iterrows():
        ax_line.text(
            row["rank"],
            row["mean_true_risk"] + 0.003,
            f"{row['mean_true_risk']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_line.set_xlabel("Segment index (1 = highest-risk)")
    ax_line.set_ylabel("P(disease_risk = 1)")
    ax_line.set_title("Disease risk rate by segment")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(loc="upper right")
    fig_line.tight_layout()
    plt.show()


    n_seg = len(seg_table)
    top_k = max(1, int(np.ceil(0.5 * n_seg)))  # 前 50%
    top_segments = seg_sorted.iloc[:top_k]

    seg_feature_means = []
    seg_ids = []

    for seg_id in top_segments["segment_id"]:
        sub = df_with_segments[df_with_segments["segment_id"] == seg_id]
        seg_feature_means.append(sub[feature_cols].mean())
        seg_ids.append(seg_id)

    heatmap_df = pd.DataFrame(
        seg_feature_means, index=seg_ids, columns=feature_cols
    )

    heatmap_norm = (heatmap_df - heatmap_df.min()) / (
        heatmap_df.max() - heatmap_df.min() + 1e-9
    )

    plt.figure(figsize=(len(feature_cols) * 0.5 + 4, top_k * 0.4 + 2))
    im = plt.imshow(
        heatmap_norm.values,
        aspect="auto",
        cmap="bwr",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label="Normalized feature value (0=blue, 1=red)")

    plt.yticks(
        ticks=np.arange(top_k),
        labels=[str(sid) for sid in heatmap_norm.index],
    )
    plt.xticks(
        ticks=np.arange(len(feature_cols)),
        labels=feature_cols,
        rotation=45,
        ha="right",
    )
    plt.xlabel("Features")
    plt.ylabel("High-risk segments (top 50%)")
    plt.title("Feature patterns of top 50% high-risk segments")
    plt.tight_layout()
    plt.show()


    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))

    sizes = seg_top["n_samples"].values
    wedges, _ = ax_pie.pie(
        sizes,
        startangle=90,
    )
    ax_pie.set_title("Population share by segment")
    ax_pie.axis("equal")

    total_n = sizes.sum()
    legend_labels = [
        f"S{int(rank)} ({n / total_n * 100:.1f}%)"
        for rank, n in zip(seg_top["rank"], sizes)
    ]

    ax_pie.legend(
        wedges,
        legend_labels,
        title="Segment (population share)",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
    )

    fig_pie.tight_layout()
    plt.show()


    seg_group = df_with_segments.groupby("segment_id")
    seg_feat_means_all = seg_group[feature_cols].mean()
    seg_risk_all = seg_group[target_col].mean()

    seg_feat_means_all["mean_true_risk"] = seg_risk_all

    corr_with_risk = seg_feat_means_all.corr()["mean_true_risk"].drop(
        "mean_true_risk"
    )

    corr_df = (
        corr_with_risk.reset_index()
        .rename(columns={"index": "feature", "mean_true_risk": "corr_with_risk"})
        .sort_values("corr_with_risk", ascending=False)
    )

    print("\n===== Feature–risk correlation at segment level =====")
    print(corr_df.to_string(index=False))

    plt.figure(figsize=(max(6, len(feature_cols) * 0.4), 4))
    plt.bar(corr_df["feature"], corr_df["corr_with_risk"])
    plt.axhline(0, linewidth=0.5)
    plt.ylabel("Correlation with segment mean risk")
    plt.xlabel("Features")
    plt.title("Feature influence on risk (segment-level correlation)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


    leaf_logit_models = train_leaf_logistic_models(
        tree_clf, X_train[feature_cols], y_train
    )


    leaf_ids_train = tree_clf.apply(X_train[feature_cols])


    valid_leaf_ids = [lid for lid, mdl in leaf_logit_models.items() if mdl is not None]
    n_plots = len(valid_leaf_ids)

    if n_plots > 0:

        n_cols = 4
        n_rows = int(np.ceil(n_plots / n_cols))

        fig_lr, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False
        )
        axes_flat = axes.ravel()

        for idx, leaf_id in enumerate(valid_leaf_ids):
            ax = axes_flat[idx]
            mdl = leaf_logit_models[leaf_id]


            mask = (leaf_ids_train == leaf_id)
            X_leaf = X_train.loc[mask, feature_cols]
            y_leaf = y_train.loc[mask]


            scores = mdl.decision_function(X_leaf)
            s_min, s_max = float(scores.min()), float(scores.max())
            if s_min == s_max:
                s_min -= 1.0
                s_max += 1.0

            s_grid = np.linspace(s_min, s_max, 200)
            p_grid = 1.0 / (1.0 + np.exp(-s_grid))

            ax.plot(s_grid, p_grid, label=f"leaf {leaf_id}")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Linear score (w^T x + b)")
            ax.set_ylabel("P(risk=1)")
            ax.set_title(f"Leaf {leaf_id}  (n={mask.sum()})")


            ax.scatter(scores, y_leaf, s=10, alpha=0.4, color="gray")


        for j in range(n_plots, len(axes_flat)):
            axes_flat[j].axis("off")

        fig_lr.suptitle("Logistic curves inside tree leaves", fontsize=14, y=1.02)
        fig_lr.tight_layout()
        plt.show()


    sample = {
        "age": 50,
        "gender_num": 1,
        "bmi": 29.0,
        "daily_steps": 5000,
        "sleep_hours": 6.5,
        "water_intake_l": 2.5,
        "calories_consumed": 2200,
        "smoker": 0,
        "alcohol": 1,
        "resting_hr": 75,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "cholesterol": 210,
        "family_history": 1,
    }

    x_new = pd.DataFrame([sample])[feature_cols]


    indiv_prob_tree = tree_clf.predict_proba(x_new)[0, 1]

    leaf_id = tree_clf.apply(x_new)[0]
    seg_info = seg_table.loc[seg_table["segment_id"] == leaf_id].iloc[0]


    logit_model = leaf_logit_models.get(leaf_id, None)
    if logit_model is not None:
        indiv_prob_logit = logit_model.predict_proba(x_new)[0, 1]
    else:
        indiv_prob_logit = float(seg_info["mean_true_risk"])

    print("\n===== 单个个体示例 =====")
    print(f"所属 segment_id = {leaf_id}")
    print(f"该 segment 的规则: {seg_info['rules']}")
    print(f"该 segment 的真实平均风险 = {seg_info['mean_true_risk']:.4f}")
    print(f"决策树叶子平均预测 P(risk=1) = {indiv_prob_tree:.4f}")
    print(f"叶子内 Logistic 回归预测 P(risk=1) = {indiv_prob_logit:.4f}")


if __name__ == "__main__":
    main()
