import numpy as np

def add_high_accuracy_features(df):
    """
    Robust feature engineering for UCI Credit Default dataset
    Works even if column names are messy
    """

    # ---------- Normalize column names ----------
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    # ---------- Auto-detect bill & payment columns ----------
    bill_cols = sorted([c for c in df.columns if c.startswith("BILL_AMT")])
    pay_amt_cols = sorted([c for c in df.columns if c.startswith("PAY_AMT")])
    pay_status_cols = sorted([c for c in df.columns if c.startswith("PAY_") and "AMT" not in c])

    if len(bill_cols) < 6 or len(pay_amt_cols) < 6:
        raise ValueError(f"""
Required columns not found.
Found BILL columns: {bill_cols}
Found PAY_AMT columns: {pay_amt_cols}
""")

    # ---------- Utilization & payment ratios ----------
    for i in range(6):
        df[f"UTIL_{i+1}"] = df[bill_cols[i]] / (df["LIMIT_BAL"] + 1)
        df[f"PAY_RATIO_{i+1}"] = df[pay_amt_cols[i]] / (df[bill_cols[i]] + 1)

    # ---------- Aggregate behaviour ----------
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["MAX_BILL_AMT"] = df[bill_cols].max(axis=1)
    df["TOTAL_PAY_AMT"] = df[pay_amt_cols].sum(axis=1)

    # ---------- Payment discipline ----------
    if pay_status_cols:
        df["AVG_DELAY"] = df[pay_status_cols].mean(axis=1)
        df["MAX_DELAY"] = df[pay_status_cols].max(axis=1)

    # ---------- Spending trend ----------
    df["BILL_TREND"] = df[bill_cols[-1]] - df[bill_cols[0]]
    
        # ---------- Handle infinities & extreme values ----------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Cap extreme values (robust scaling trick)
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].clip(lower=df[col].quantile(0.01),
                                    upper=df[col].quantile(0.99))

    df = df.fillna(0)

    return df
