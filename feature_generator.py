import pandas as pd

# ==========================
# 設定
# ==========================

INPUT_CSV = "loto7.csv"
OUTPUT_CSV = "features_loto7.csv"

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

# ==========================
# 読込
# ==========================

df = pd.read_csv(INPUT_CSV)

print(f"records = {len(df)}")

# ==========================
# 出現フラグ作成
# ==========================

for n in range(1, NUMBER_MAX + 1):

    df[f"hit_{n}"] = (
        df[NUMBER_COLUMNS]
        .eq(n)
        .any(axis=1)
        .astype(int)
    )

print("hit columns created")

# ==========================
# 特徴量生成
# ==========================

rows = []

for number in range(1, NUMBER_MAX + 1):

    hit_col = f"hit_{number}"

    last_hit_index = None

    for draw_idx in range(len(df)):

        target = df.loc[draw_idx, hit_col]

        # ------------------
        # gap
        # ------------------

        if last_hit_index is None:
            gap = draw_idx + 1
        else:
            gap = draw_idx - last_hit_index
        # ------------------
        # prev_hit
        # ------------------

        if draw_idx == 0:
            prev_hit = 0
        else:
            prev_hit = df.loc[
                draw_idx - 1,
                hit_col
            ]

        # ------------------
        # 過去出現回数
        # ------------------

        start5 = max(0, draw_idx - 5)
        start10 = max(0, draw_idx - 10)
        start20 = max(0, draw_idx - 20)

        last5_count = (
            df.loc[
                start5:draw_idx - 1,
                hit_col
            ].sum()
            if draw_idx > 0
            else 0
        )

        last10_count = (
            df.loc[
                start10:draw_idx - 1,
                hit_col
            ].sum()
            if draw_idx > 0
            else 0
        )

        last20_count = (
            df.loc[
                start20:draw_idx - 1,
                hit_col
            ].sum()
            if draw_idx > 0
            else 0
        )

        # ------------------
        # レコード生成
        # ------------------

        rows.append(
            {
                "draw_no": draw_idx + 1,
                "number": number,

                # 目的変数
                "target": target,

                # 時系列特徴量
                "gap": gap,
                "prev_hit": prev_hit,
                "last5_count": last5_count,
                "last10_count": last10_count,
                "last20_count": last20_count,

                # 気象
                "temp": df.loc[draw_idx, "temp"],
                "temp_diff": df.loc[
                    draw_idx,
                    "temp_diff"
                ],

                "pressure": df.loc[
                    draw_idx,
                    "pressure"
                ],

                "pressure_diff": df.loc[
                    draw_idx,
                    "pressure_diff"
                ],

                "humidity": df.loc[
                    draw_idx,
                    "humidity"
                ],

                "humidity_diff": df.loc[
                    draw_idx,
                    "humidity_diff"
                ],

                # 月齢
                "moon_age": df.loc[
                    draw_idx,
                    "moon_age"
                ],

                "moon_sin": df.loc[
                    draw_idx,
                    "moon_sin"
                ],

                "moon_cos": df.loc[
                    draw_idx,
                    "moon_cos"
                ]
            }
        )

        # ------------------
        # 更新
        # ------------------

        if target == 1:
            last_hit_index = draw_idx

feature_df = pd.DataFrame(rows)

feature_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print()
print(f"saved: {OUTPUT_CSV}")
print()

print(feature_df.head())

print()
print("shape =", feature_df.shape)

print()
print("===== GAP ANALYSIS =====")

print(
    feature_df.groupby("gap")["target"]
    .mean()
    .head(30)
)
print()
print("===== LAST5 ANALYSIS =====")

print(
    feature_df.groupby("last5_count")["target"]
    .mean()
)

print()
print("===== LAST10 ANALYSIS =====")

print(
    feature_df.groupby("last10_count")["target"]
    .mean()
)