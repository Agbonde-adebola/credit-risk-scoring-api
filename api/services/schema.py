import pandas as pd

RAW_REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

RAW_ALLOWED_HOME_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "OTHER"}

RAW_ALLOWED_LOAN_INTENT = {
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
}

RAW_ALLOWED_DEFAULT_ON_FILE = {"Y", "N"}

RAW_ALLOWED_LOAN_GRADE = {"A", "B", "C", "D", "E", "F", "G"}


def _normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def validate_raw_schema(df_raw: pd.DataFrame) -> None:
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("Raw input must be a pandas DataFrame")

    # Required columns
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(
            "Missing required raw input columns: " + ", ".join(missing)
        )

    # person_home_ownership
    bad_home = set(_normalize(
        df_raw["person_home_ownership"])) - RAW_ALLOWED_HOME_OWNERSHIP
    if bad_home:
        raise ValueError(
            f"Invalid person_home_ownership values: {sorted(bad_home)}. "
            f"Allowed: {sorted(RAW_ALLOWED_HOME_OWNERSHIP)}"
        )

    # loan_intent
    bad_intent = set(_normalize(
        df_raw["loan_intent"])) - RAW_ALLOWED_LOAN_INTENT
    if bad_intent:
        raise ValueError(
            f"Invalid loan_intent values: {sorted(bad_intent)}. "
            f"Allowed: {sorted(RAW_ALLOWED_LOAN_INTENT)}"
        )

    # cb_person_default_on_file
    bad_default = set(_normalize(
        df_raw["cb_person_default_on_file"])) - RAW_ALLOWED_DEFAULT_ON_FILE
    if bad_default:
        raise ValueError(
            f"Invalid cb_person_default_on_file values: {sorted(bad_default)}. "
            f"Allowed: {sorted(RAW_ALLOWED_DEFAULT_ON_FILE)}"
        )

    # loan_grade
    bad_grade = set(_normalize(df_raw["loan_grade"])) - RAW_ALLOWED_LOAN_GRADE
    if bad_grade:
        raise ValueError(
            f"Invalid loan_grade values: {sorted(bad_grade)}. "
            f"Allowed: {sorted(RAW_ALLOWED_LOAN_GRADE)}"
        )
