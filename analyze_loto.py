import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from itertools import combinations
from statsmodels.stats.multitest import multipletests


# ==========================
# 設定
# ==========================

CSV_FILE = "loto7.csv"

# Loto7
NUMBER_MAX = 37
NUMBER_COLUMNS = [
    "num1",
    "num2",
    "num3",
    "num4",
    "num5",
    "num6",
    "num7"
]

# Loto6の場合
# NUMBER_MAX = 43
# NUMBER_COLUMNS = [
#     "num1",
#     "num2",
#     "num3",
#     "num4",
#     "num5",
#     "num6"
# ]

FEATURES = [
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
# CSV読込
# ==========================

df = pd.read_csv(CSV_FILE)

print(f"records = {len(df)}")

# ==========================
# 出現フラグ作成
# ==========================

for n in range(1, NUMBER_MAX + 1):

    df[f"hit_{n}"] = df[NUMBER_COLUMNS].eq(n).any(axis=1).astype(int)

print("hit columns created")

# ==========================
# 相関分析
# ==========================

corr_results = []

for number in range(1, NUMBER_MAX + 1):

    target = df[f"hit_{number}"]

    for feature in FEATURES:

        try:

            corr, p = pointbiserialr(target, df[feature])

            corr_results.append(
                {
                    "number": number,
                    "feature": feature,
                    "corr": corr,
                    "p_value": p
                }
            )

        except Exception:

            pass

corr_df = pd.DataFrame(corr_results)

corr_df["abs_corr"] = corr_df["corr"].abs()


corr_df["p_adj"] = multipletests(
    corr_df["p_value"],
    method="fdr_bh"
)[1]


corr_df = corr_df.sort_values(
    by="p_adj"
)

corr_df.to_csv(
    "correlation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("correlation_results.csv saved")

# ==========================
# Lift分析
# ==========================

lift_results = []

for n1, n2 in combinations(
        range(1, NUMBER_MAX + 1), 2):

    hit1 = df[f"hit_{n1}"]
    hit2 = df[f"hit_{n2}"]

    p1 = hit1.mean()
    p2 = hit2.mean()

    both_mask = (
        (hit1 == 1) &
        (hit2 == 1)
    )

    both_count = both_mask.sum()

    p_both = both_count / len(df)

    expected = p1 * p2

    if expected == 0:
        continue

    lift = p_both / expected

    lift_results.append(
        {
            "num1": n1,
            "num2": n2,
            "lift": lift,
            "joint_prob": p_both,
            "both_count": both_count,
            "expected_prob": expected
        }
    )

lift_df = pd.DataFrame(lift_results)

# 共起回数が少なすぎるものを除外
lift_df = lift_df[
    lift_df["both_count"] >= 10
]

lift_df = lift_df.sort_values(
    by=["lift", "both_count"],
    ascending=[False, False]
)

lift_df.to_csv(
    "lift_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("lift_results.csv saved")

print()
print("===== TOP20 LIFT =====")

print(
    lift_df[
        [
            "num1",
            "num2",
            "lift",
            "both_count",
            "joint_prob"
        ]
    ].head(20)
)
# ==========================
# 出現頻度
# ==========================

freq_results = []

for n in range(1, NUMBER_MAX + 1):

    count = df[f"hit_{n}"].sum()

    freq_results.append(
        {
            "number": n,
            "count": count,
            "rate": count / len(df)
        }
    )

freq_df = pd.DataFrame(freq_results)

freq_df = freq_df.sort_values(
    by="count",
    ascending=False
)

freq_df.to_csv(
    "frequency_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("frequency_results.csv saved")

print()
print("===== TOP10 FREQUENCY =====")

print(freq_df.head(10))

print()
print("===== TOP10 CORRELATION =====")

print(
    corr_df[
        [
            "number",
            "feature",
            "corr",
            "p_value",
            "p_adj"
        ]
    ].head(10)
)
print()
print("===== TOP10 LIFT =====")

print(
    lift_df[
        ["num1",
         "num2",
         "lift"]
    ].head(10)
)
print()
print("===== BOTTOM10 FREQUENCY =====")

print(freq_df.tail(10))

from scipy.stats import chisquare

counts = []

for n in range(1, NUMBER_MAX + 1):
    counts.append(
        df[f"hit_{n}"].sum()
    )

chi2, p = chisquare(counts)

print()
print("===== FREQUENCY CHI-SQUARE =====")
print(f"chi2={chi2:.4f}")
print(f"p={p:.6f}")