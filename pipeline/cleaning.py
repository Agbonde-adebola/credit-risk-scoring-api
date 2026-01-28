import pandas as pd


def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Remove pandas index artifacts if present
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    # Trim/normalize string fields
    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # Numeric coercion
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic sanity handling (light-touch)
    df["person_age"] = df["person_age"].clip(lower=18, upper=100)
    df["person_income"] = df["person_income"].clip(lower=0)
    df["person_emp_length"] = df["person_emp_length"].clip(lower=0, upper=60)
    df["loan_amnt"] = df["loan_amnt"].clip(lower=0)
    df["loan_int_rate"] = df["loan_int_rate"].clip(lower=0, upper=100)
    df["loan_percent_income"] = df["loan_percent_income"].clip(lower=0, upper=5)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(lower=0, upper=80)

    # Missing numeric values -> median per column
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df
