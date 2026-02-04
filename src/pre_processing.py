import pandas as pd

def load_and_clean_data(file_path):

    #  IMPORTANT FIX: header is on row 1
    df = pd.read_csv(file_path, header=1)

    # ---------- Normalize column names ----------
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    # ---------- Drop ID ----------
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # ---------- Fix categorical anomalies ----------
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # ---------- Convert everything to numeric ----------
    df = df.apply(pd.to_numeric, errors="coerce")

    # ---------- Handle missing ----------
    df = df.fillna(0)

    return df
