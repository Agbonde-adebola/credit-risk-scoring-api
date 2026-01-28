import pandas as pd


def _encode_loan_grade(series: pd.Series) -> pd.Series:
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    return series.map(mapping).fillna(0).astype(int)


def engineer_features(df_clean: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    df = df_clean.copy()

    # Start with a features frame
    feats = pd.DataFrame(index=df.index)

    # Numeric direct features (common in this dataset)
    feats["Age"] = df["person_age"].astype(float)
    feats["Annual Income"] = df["person_income"].astype(float)
    feats["Employment Length (years)"] = df["person_emp_length"].astype(float)
    feats["Loan Amount"] = df["loan_amnt"].astype(float)
    feats["Interest Rate"] = df["loan_int_rate"].astype(float)
    feats["Loan to Income Ratio"] = df["loan_percent_income"].astype(float)
    feats["Credit History Length (years)"] = df["cb_person_cred_hist_length"].astype(float)

    # cb_person_default_on_file -> Has Defaulted_N / Has Defaulted_Y
    flag = df["cb_person_default_on_file"].astype(str).str.strip().str.upper()
    feats["Has Defaulted_Y"] = (flag == "Y").astype(int)
    feats["Has Defaulted_N"] = (flag == "N").astype(int)

    # One-hot: Home Ownership_*
    home = df["person_home_ownership"].astype(str).str.strip().str.upper()
    for cat in ["MORTGAGE", "OTHER", "OWN", "RENT"]:
        feats[f"Home Ownership_{cat}"] = (home == cat).astype(int)

    # One-hot: Loan Intent_*
    intent = df["loan_intent"].astype(str).str.strip().str.upper()
    for cat in [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ]:
        feats[f"Loan Intent_{cat}"] = (intent == cat).astype(int)

    # Ordinal: loan_grade -> loan_grade_encoded
    feats["loan_grade_encoded"] = _encode_loan_grade(df["loan_grade"].astype(str).str.strip().str.upper())

    # Make sure features exactly match expected schema and order
    feats = feats.reindex(columns=expected_features, fill_value=0)

    return feats
