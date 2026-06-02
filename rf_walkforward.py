import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# ==========================
# 設定
# ==========================

INPUT_FILE = "features_loto7.csv"

TRAIN_START = 200   # 学習開始に必要な最低回数

N_TREES = 200

FEATURES = [
    "gap",
    "prev_hit",
    "last5_count",
    "last10_count",
    "last20_count",
    "temp",
    "temp_diff",
    "pressure",
    "pressure_diff",
    "humidity",
    "humidity_diff",
    "moon_age",
    "moon_sin",
    "moon_cos"
]

# ==========================
# 読込
# ==========================

df = pd.read_csv(INPUT_FILE)

max_draw = df["draw_no"].max()

print(f"records={len(df)}")
print(f"draws={max_draw}")

# ==========================
# ウォークフォワード
# ==========================

results = []

for target_draw in range(
        TRAIN_START + 1,
        max_draw + 1):

    train_df = df[
        df["draw_no"] < target_draw
    ]

    test_df = df[
        df["draw_no"] == target_draw
    ]

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    pred_df = test_df.copy()
    pred_df["prob"] = probs

    pred_df = pred_df.sort_values(
        by="prob",
        ascending=False
    )

    # 上位7個を予測数字とする
    predicted_numbers = set(
        pred_df.head(7)["number"]
    )

    actual_numbers = set(
        test_df[
            test_df["target"] == 1
        ]["number"]
    )

    hit_count = len(
        predicted_numbers &
        actual_numbers
    )

    results.append(
        {
            "draw_no": target_draw,
            "hit_count": hit_count
        }
    )

    if target_draw % 50 == 0:
        print(
            f"draw={target_draw} "
            f"hit={hit_count}"
        )

# ==========================
# 集計
# ==========================

result_df = pd.DataFrame(results)

print()
print("===== RESULT =====")

print(result_df["hit_count"].describe())

avg_hit = result_df["hit_count"].mean()

print()
print(
    f"Average hit count = "
    f"{avg_hit:.4f}"
)

# ==========================
# ランダム予測期待値
# ==========================

random_expectation = 7 * 7 / 37

print(
    f"Random expectation = "
    f"{random_expectation:.4f}"
)

print(
    f"Difference = "
    f"{avg_hit - random_expectation:.4f}"
)

result_df.to_csv(
    "rf_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("saved: rf_result.csv")

# ==========================
# Feature Importance
# ==========================

print()
print("===== FEATURE IMPORTANCE =====")

final_train = df[
    df["draw_no"] < max_draw
]

X = final_train[FEATURES]
y = final_train["target"]

final_model = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

final_model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df)

importance_df.to_csv(
    "feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("saved: feature_importance.csv")