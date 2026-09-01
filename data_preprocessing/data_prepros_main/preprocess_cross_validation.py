# preprocess_cross_validation.py

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# Column groups
# ============================================================
TARGET_COLUMN = "Rec_DGF"
ID_COLUMN = "Rec_code"

CONTINUOUS_NUMERIC_COLUMNS = [
    "Rec_Age",
    "Rec_BMI",
    "Do_Age",
    "Do_BMI",
    "Do_KDRI",
    "Do_Cold_Ischemia_Min",
    "Do_Warm_Ischemia_Min",
    "Do_Total_Ischemia_Min",
    "Rec_Hb",
    "Rec_Na",
    "Rec_k",
    "Rec_ca",
]

SKEWED_NUMERIC_COLUMNS = [
    "Rec_DIALYSIS_DURATION",
    "Do_Cr",
    "Rec_Cr_Day1",
    "Rec_Cr_Day3",
    "Rec_Cr_Day5",
    "Rec_Cr_Day6",
    "Rec_Cr_Day7",
    "Rec_Cr(last_day)",
    "Rec_Urine_Output",
]

PERCENT_COLUMNS = [
    "Do_KDPI",
    "Rec_PRA",
]

BINARY_COLUMNS = [
    "Rec_Sex",
    "Rec_Preemptive",
    "Rec_DM",
    "Rec_HTN",
    "Rec_CVD",
    "Do_Sex",
    "Do_Type",
    "Bo_Gender_Match",
    "HLA_DRB3_Binary_RE",
    "HLA_DRB4_Binary_RE",
    "HLA_DRB5_Binary_RE",
    "HLA_DPB1_Binary_RE",
    "HLA_BW4_Binary_RE",
    "HLA_BW6_Binary_RE",
]

CATEGORICAL_COLUMNS = [
    "Rec_ESRD_Cause",
    "Rec_Blood_Type",
    "Do_Cause_of_Death",
    "Do_Blood_Type",
]

HLA_COUNT_COLUMNS = [
    "HLA_A_Count_RE",
    "HLA_B_Count_RE",
    "HLA_C_Count_RE",
    "HLA_DRB1_Count_RE",
    "HLA_DQB1_Count_RE",
    "HLA_DQA1_Count_RE",
]

ESRD_CATEGORIES = [
    "HTN",
    "Proteinuria_Glomerular",
    "DM",
    "PKD",
    "Infection_Reflux_Urologic",
    "Lupus_SLE",
    "Congenital_Urologic",
    "Stone_Urologic",
    "Unknown",
]

CAUSE_OF_DEATH_CATEGORIES = [
    "Cerebrovascular_Hemorrhage",
    "Trauma",
    "Poisoning_Overdose",
    "Other_CNS_Neurologic",
    "Anoxic_Hypoxic_Asphyxia_CPR",
    "Unknown_Not_Applicable",
    "Other_Medical",
    "Not_Applicable_Living_Donor",
]

BLOOD_TYPE_CATEGORIES = [
    "A",
    "B",
    "AB",
    "O",
]

MISSING_TOKENS = {
    "",
    " ",
    "-",
    "–",
    "—",
    "−",
    "?",
    "؟",
    ".",
    "..",
    "...",
    "nan",
    "NaN",
    "NAN",
    "none",
    "None",
    "NONE",
    "null",
    "NULL",
    "#DIV/0!",
    "#VALUE!",
    "#REF!",
    "#N/A",
    "#NAME?",
    "#NUM!",
    "#NULL!",
    "خالی",
    "نامشخص",
    "نامعلوم",
    "مشخص نیست",
}


# ============================================================
# Basic cleaning functions
# ============================================================
def clean_column_names(df):
    """
    Clean column names.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def normalize_digits(value):
    """
    Convert Persian and Arabic digits to English digits.
    """
    if pd.isna(value):
        return value

    text = str(value)

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"

    for p, e in zip(persian_digits, english_digits):
        text = text.replace(p, e)

    for a, e in zip(arabic_digits, english_digits):
        text = text.replace(a, e)

    return text


def is_missing_like(value):
    """
    Detect missing-like values.
    """
    if pd.isna(value):
        return True

    value_str = normalize_digits(value)
    value_str = str(value_str).strip()

    if value_str in MISSING_TOKENS:
        return True

    return False


def clean_numeric_series(series):
    """
    Convert a series to numeric values.
    """
    cleaned = series.copy()

    cleaned = cleaned.apply(lambda x: np.nan if is_missing_like(x) else normalize_digits(x))
    cleaned = cleaned.astype(str).str.strip()
    cleaned = cleaned.str.replace(",", ".", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def clean_binary_series(series):
    """
    Convert binary values to 0 and 1.
    """
    def convert_value(value):
        if is_missing_like(value):
            return np.nan

        value_str = normalize_digits(value)
        value_str = str(value_str).strip().lower()

        one_values = {
            "1",
            "1.0",
            "yes",
            "y",
            "true",
            "male",
            "m",
            "مرد",
            "بله",
            "بلی",
            "دارد",
            "مثبت",
            "positive",
            "deceased",
        }

        zero_values = {
            "0",
            "0.0",
            "no",
            "n",
            "false",
            "female",
            "f",
            "زن",
            "خیر",
            "نه",
            "ندارد",
            "منفی",
            "negative",
            "living",
        }

        if value_str in one_values:
            return 1

        if value_str in zero_values:
            return 0

        return np.nan

    return series.apply(convert_value)


def clean_text_value(value):
    """
    Clean a text value.
    """
    if is_missing_like(value):
        return np.nan

    text = normalize_digits(value)
    text = str(text).strip()

    text = text.replace("\u200c", "")
    text = text.replace("\u200f", "")
    text = text.replace("\u200e", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", "_", text)

    return text


# ============================================================
# Categorical normalizers
# ============================================================
def normalize_blood_type(value):
    """
    Normalize blood type to A, B, AB, or O.
    """
    value = clean_text_value(value)

    if pd.isna(value):
        return np.nan

    value = str(value).upper()
    value = value.replace("+", "")
    value = value.replace("-", "")
    value = value.replace("_", "")
    value = value.strip()

    if value in BLOOD_TYPE_CATEGORIES:
        return value

    return np.nan


def normalize_esrd_cause(value):
    """
    Normalize ESRD cause.
    DM_HTN is kept temporarily and will be converted to both DM and HTN in custom one-hot encoding.
    """
    value = clean_text_value(value)

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    mapping = {
        "HTN": "HTN",
        "Proteinuria_Glomerular": "Proteinuria_Glomerular",
        "Glomerular_Proteinuria": "Proteinuria_Glomerular",
        "DM": "DM",
        "PKD": "PKD",
        "Infection_Reflux_Urologic": "Infection_Reflux_Urologic",
        "Lupus_SLE": "Lupus_SLE",
        "SLE_Lupus": "Lupus_SLE",
        "DM_HTN": "DM_HTN",
        "HTN_DM": "DM_HTN",
        "Congenital_Urologic": "Congenital_Urologic",
        "Stone_Urologic": "Stone_Urologic",
        "Unknown": "Unknown",
        "Unknown_Not_Applicable": "Unknown",
    }

    return mapping.get(value, "Unknown")


def normalize_cause_of_death(value):
    """
    Normalize donor cause of death.
    """
    value = clean_text_value(value)

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    mapping = {
        "Cerebrovascular_Hemorrhage": "Cerebrovascular_Hemorrhage",
        "Trauma": "Trauma",
        "Poisoning_Overdose": "Poisoning_Overdose",
        "Other_CNS_Neurologic": "Other_CNS_Neurologic",
        "Anoxic_Hypoxic_Asphyxia_CPR": "Anoxic_Hypoxic_Asphyxia_CPR",
        "Unknown_Not_Applicable": "Unknown_Not_Applicable",
        "Other_Medical": "Other_Medical",
        "Not_Applicable_Living_Donor": "Not_Applicable_Living_Donor",
    }

    return mapping.get(value, "Unknown_Not_Applicable")


# ============================================================
# Pre-fit utilities
# ============================================================
def safe_std(series):
    """
    Return standard deviation. If zero or NaN, return 1.
    """
    std_value = series.std(ddof=0)

    if pd.isna(std_value) or std_value == 0:
        return 1.0

    return std_value


def get_mode(series, default_value):
    """
    Return mode of a series. If empty, return default value.
    """
    values = series.dropna()

    if len(values) == 0:
        return default_value

    return values.mode().iloc[0]


def create_missing_indicators(df, columns):
    """
    Create missing indicator columns.
    """
    indicator_df = pd.DataFrame(index=df.index)

    for col in columns:
        if col in df.columns:
            numeric_col = clean_numeric_series(df[col])
            indicator_df[f"{col}_missing"] = numeric_col.isna().astype(int)

    return indicator_df


# ============================================================
# Dataset preparation before CV
# ============================================================
def prepare_dataframe_before_cv(df):
    """
    Apply row-wise rules before cross-validation.
    """
    df = df.copy()

    if "Rec_ESRD_Cause" in df.columns:
        esrd_clean = df["Rec_ESRD_Cause"].apply(normalize_esrd_cause)

        dm_htn_mask = esrd_clean == "DM_HTN"

        if "Rec_DM" in df.columns:
            df.loc[dm_htn_mask, "Rec_DM"] = 1

        if "Rec_HTN" in df.columns:
            df.loc[dm_htn_mask, "Rec_HTN"] = 1

        df["Rec_ESRD_Cause"] = esrd_clean

    if "Rec_Blood_Type" in df.columns:
        df["Rec_Blood_Type"] = df["Rec_Blood_Type"].apply(normalize_blood_type)

    if "Do_Blood_Type" in df.columns:
        df["Do_Blood_Type"] = df["Do_Blood_Type"].apply(normalize_blood_type)

    if "Do_Cause_of_Death" in df.columns:
        df["Do_Cause_of_Death"] = df["Do_Cause_of_Death"].apply(normalize_cause_of_death)

    if "Do_Type" in df.columns and "Do_Cause_of_Death" in df.columns:
        do_type_clean = clean_binary_series(df["Do_Type"])
        cod_missing = df["Do_Cause_of_Death"].isna()

        living_mask = do_type_clean == 0
        df.loc[living_mask & cod_missing, "Do_Cause_of_Death"] = "Not_Applicable_Living_Donor"

    return df


# ============================================================
# Fit preprocessing on train fold only
# ============================================================
def fit_preprocessor(X_train):
    """
    Fit imputers and scalers on train fold only.
    """
    state = {}

    existing_continuous = [c for c in CONTINUOUS_NUMERIC_COLUMNS if c in X_train.columns]
    existing_skewed = [c for c in SKEWED_NUMERIC_COLUMNS if c in X_train.columns]
    existing_percent = [c for c in PERCENT_COLUMNS if c in X_train.columns]
    existing_binary = [c for c in BINARY_COLUMNS if c in X_train.columns]
    existing_hla_count = [c for c in HLA_COUNT_COLUMNS if c in X_train.columns]

    state["existing_continuous"] = existing_continuous
    state["existing_skewed"] = existing_skewed
    state["existing_percent"] = existing_percent
    state["existing_binary"] = existing_binary
    state["existing_hla_count"] = existing_hla_count

    state["continuous"] = {}
    for col in existing_continuous:
        s = clean_numeric_series(X_train[col])
        median_value = s.median()
        s_filled = s.fillna(median_value)

        state["continuous"][col] = {
            "median": median_value,
            "mean": s_filled.mean(),
            "std": safe_std(s_filled),
        }

    state["skewed"] = {}
    for col in existing_skewed:
        s = clean_numeric_series(X_train[col])
        median_value = s.median()
        s_filled = s.fillna(median_value)
        s_filled = s_filled.clip(lower=0)
        s_log = np.log1p(s_filled)

        state["skewed"][col] = {
            "median": median_value,
            "mean": s_log.mean(),
            "std": safe_std(s_log),
        }

    state["percent"] = {}
    for col in existing_percent:
        s = clean_numeric_series(X_train[col])
        median_value = s.median()
        s_filled = s.fillna(median_value) / 100.0

        state["percent"][col] = {
            "median": median_value,
            "mean": s_filled.mean(),
            "std": safe_std(s_filled),
        }

    state["binary"] = {}
    for col in existing_binary:
        s = clean_binary_series(X_train[col])
        mode_value = get_mode(s, default_value=0)

        state["binary"][col] = {
            "mode": mode_value,
        }

    state["hla_count"] = {}
    for col in existing_hla_count:
        s = clean_numeric_series(X_train[col])
        median_value = s.median()
        s_filled = s.fillna(median_value)

        state["hla_count"][col] = {
            "median": median_value,
            "mean": s_filled.mean(),
            "std": safe_std(s_filled),
        }

    state["categorical_modes"] = {}

    if "Rec_ESRD_Cause" in X_train.columns:
        s = X_train["Rec_ESRD_Cause"].apply(normalize_esrd_cause)
        state["categorical_modes"]["Rec_ESRD_Cause"] = get_mode(s, default_value="Unknown")

    if "Rec_Blood_Type" in X_train.columns:
        s = X_train["Rec_Blood_Type"].apply(normalize_blood_type)
        state["categorical_modes"]["Rec_Blood_Type"] = get_mode(s, default_value="O")

    if "Do_Cause_of_Death" in X_train.columns:
        s = X_train["Do_Cause_of_Death"].apply(normalize_cause_of_death)
        state["categorical_modes"]["Do_Cause_of_Death"] = get_mode(
            s,
            default_value="Unknown_Not_Applicable"
        )

    if "Do_Blood_Type" in X_train.columns:
        s = X_train["Do_Blood_Type"].apply(normalize_blood_type)
        state["categorical_modes"]["Do_Blood_Type"] = get_mode(s, default_value="O")

    return state


# ============================================================
# Transform train or validation fold
# ============================================================
def transform_with_preprocessor(X, state):
    """
    Transform a dataframe using fitted preprocessing state.
    """
    X = X.copy()
    output_parts = []

    for col in state["existing_continuous"]:
        s = clean_numeric_series(X[col])
        params = state["continuous"][col]

        s = s.fillna(params["median"])
        s_scaled = (s - params["mean"]) / params["std"]

        output_parts.append(pd.DataFrame({col: s_scaled}, index=X.index))

    for col in state["existing_skewed"]:
        s = clean_numeric_series(X[col])
        missing_indicator = s.isna().astype(int)

        params = state["skewed"][col]

        s = s.fillna(params["median"])
        s = s.clip(lower=0)
        s_log = np.log1p(s)
        s_scaled = (s_log - params["mean"]) / params["std"]

        output_parts.append(
            pd.DataFrame(
                {
                    col: s_scaled,
                    f"{col}_missing": missing_indicator,
                },
                index=X.index
            )
        )

    for col in state["existing_percent"]:
        s = clean_numeric_series(X[col])
        params = state["percent"][col]

        s = s.fillna(params["median"])
        s = s / 100.0
        s_scaled = (s - params["mean"]) / params["std"]

        output_parts.append(pd.DataFrame({col: s_scaled}, index=X.index))

    for col in state["existing_binary"]:
        s = clean_binary_series(X[col])
        params = state["binary"][col]

        s = s.fillna(params["mode"])

        output_parts.append(pd.DataFrame({col: s.astype(int)}, index=X.index))

    for col in state["existing_hla_count"]:
        s = clean_numeric_series(X[col])
        missing_indicator = s.isna().astype(int)

        params = state["hla_count"][col]

        s = s.fillna(params["median"])
        s_scaled = (s - params["mean"]) / params["std"]

        output_parts.append(
            pd.DataFrame(
                {
                    col: s_scaled,
                    f"{col}_missing": missing_indicator,
                },
                index=X.index
            )
        )

    if "Rec_ESRD_Cause" in X.columns:
        s = X["Rec_ESRD_Cause"].apply(normalize_esrd_cause)
        fill_value = state["categorical_modes"].get("Rec_ESRD_Cause", "Unknown")
        s = s.fillna(fill_value)

        esrd_encoded = pd.DataFrame(index=X.index)

        for cat in ESRD_CATEGORIES:
            esrd_encoded[f"Rec_ESRD_Cause_{cat}"] = 0

        for idx, value in s.items():
            if value == "DM_HTN":
                esrd_encoded.loc[idx, "Rec_ESRD_Cause_DM"] = 1
                esrd_encoded.loc[idx, "Rec_ESRD_Cause_HTN"] = 1
            elif value in ESRD_CATEGORIES:
                esrd_encoded.loc[idx, f"Rec_ESRD_Cause_{value}"] = 1
            else:
                esrd_encoded.loc[idx, "Rec_ESRD_Cause_Unknown"] = 1

        output_parts.append(esrd_encoded)

    if "Rec_Blood_Type" in X.columns:
        s = X["Rec_Blood_Type"].apply(normalize_blood_type)
        fill_value = state["categorical_modes"].get("Rec_Blood_Type", "O")
        s = s.fillna(fill_value)

        blood_encoded = pd.DataFrame(index=X.index)

        for cat in BLOOD_TYPE_CATEGORIES:
            blood_encoded[f"Rec_Blood_Type_{cat}"] = (s == cat).astype(int)

        output_parts.append(blood_encoded)

    if "Do_Cause_of_Death" in X.columns:
        s = X["Do_Cause_of_Death"].apply(normalize_cause_of_death)
        fill_value = state["categorical_modes"].get(
            "Do_Cause_of_Death",
            "Unknown_Not_Applicable"
        )
        s = s.fillna(fill_value)

        cod_encoded = pd.DataFrame(index=X.index)

        for cat in CAUSE_OF_DEATH_CATEGORIES:
            cod_encoded[f"Do_Cause_of_Death_{cat}"] = (s == cat).astype(int)

        output_parts.append(cod_encoded)

    if "Do_Blood_Type" in X.columns:
        s = X["Do_Blood_Type"].apply(normalize_blood_type)
        fill_value = state["categorical_modes"].get("Do_Blood_Type", "O")
        s = s.fillna(fill_value)

        blood_encoded = pd.DataFrame(index=X.index)

        for cat in BLOOD_TYPE_CATEGORIES:
            blood_encoded[f"Do_Blood_Type_{cat}"] = (s == cat).astype(int)

        output_parts.append(blood_encoded)

    if len(output_parts) == 0:
        return pd.DataFrame(index=X.index)

    X_processed = pd.concat(output_parts, axis=1)

    return X_processed


# ============================================================
# Target cleaning
# ============================================================
def clean_target_series(series):
    """
    Clean target column and convert it to 0/1.
    """
    return clean_binary_series(series)


def remove_missing_target_rows(df, target_col=TARGET_COLUMN, id_col=ID_COLUMN):
    """
    Remove rows with missing target values.
    """
    df = df.copy()

    y_clean = clean_target_series(df[target_col])
    missing_mask = y_clean.isna()

    removed_rows = df.loc[missing_mask, [id_col, target_col]].copy()
    removed_rows["Original_Index"] = removed_rows.index

    print("\n============================================")
    print("Target cleaning")
    print("============================================")
    print(f"Target column: {target_col}")
    print(f"Total rows before target cleaning: {len(df)}")
    print(f"Rows removed because target is missing: {missing_mask.sum()}")

    if len(removed_rows) > 0:
        print("\nRemoved rows:")
        print(removed_rows.to_string(index=False))
    else:
        print("\nNo rows were removed.")

    df_cleaned = df.loc[~missing_mask].copy()
    df_cleaned[target_col] = y_clean.loc[~missing_mask].astype(int)

    print("\nTarget distribution after cleaning:")
    print(df_cleaned[target_col].value_counts(dropna=False))
    print("============================================")

    return df_cleaned, removed_rows


# ============================================================
# Main CV preprocessing function
# ============================================================
def run_cv_preprocessing(
    input_file,
    output_folder,
    target_col=TARGET_COLUMN,
    id_col=ID_COLUMN,
    header_row=0,
    n_splits=5,
    random_state=42
):
    """
    Run 5-fold cross-validation preprocessing and save processed folds.
    """
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(input_file, header=header_row, dtype=object)
    df = clean_column_names(df)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}"
        )

    if id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found. Available columns: {df.columns.tolist()}"
        )

    df = prepare_dataframe_before_cv(df)

    df_cleaned, removed_rows = remove_missing_target_rows(
        df=df,
        target_col=target_col,
        id_col=id_col
    )

    all_requested_features = (
        CONTINUOUS_NUMERIC_COLUMNS
        + SKEWED_NUMERIC_COLUMNS
        + PERCENT_COLUMNS
        + BINARY_COLUMNS
        + CATEGORICAL_COLUMNS
        + HLA_COUNT_COLUMNS
    )

    existing_features = [c for c in all_requested_features if c in df_cleaned.columns]
    missing_features = [c for c in all_requested_features if c not in df_cleaned.columns]

    if len(missing_features) > 0:
        print("\nWarning: These requested columns were not found and will be ignored:")
        for col in missing_features:
            print(f"- {col}")

    X = df_cleaned[existing_features].copy()
    y = df_cleaned[target_col].astype(int).copy()
    patient_ids = df_cleaned[id_col].copy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_summary_rows = []

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\nProcessing fold {fold_number}/{n_splits}")

        fold_folder = os.path.join(output_folder, f"fold_{fold_number}")
        os.makedirs(fold_folder, exist_ok=True)

        X_train_raw = X.iloc[train_idx].copy()
        X_valid_raw = X.iloc[valid_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        train_ids = patient_ids.iloc[train_idx].copy()
        valid_ids = patient_ids.iloc[valid_idx].copy()

        state = fit_preprocessor(X_train_raw)

        X_train_processed = transform_with_preprocessor(X_train_raw, state)
        X_valid_processed = transform_with_preprocessor(X_valid_raw, state)

        train_processed_with_id = X_train_processed.copy()
        train_processed_with_id.insert(0, id_col, train_ids.values)
        train_processed_with_id.insert(1, target_col, y_train.values)

        valid_processed_with_id = X_valid_processed.copy()
        valid_processed_with_id.insert(0, id_col, valid_ids.values)
        valid_processed_with_id.insert(1, target_col, y_valid.values)

        train_original = df_cleaned.iloc[train_idx].copy()
        valid_original = df_cleaned.iloc[valid_idx].copy()

        train_processed_path = os.path.join(fold_folder, "train_processed.xlsx")
        valid_processed_path = os.path.join(fold_folder, "valid_processed.xlsx")
        train_original_path = os.path.join(fold_folder, "train_original.xlsx")
        valid_original_path = os.path.join(fold_folder, "valid_original.xlsx")

        train_processed_with_id.to_excel(train_processed_path, index=False)
        valid_processed_with_id.to_excel(valid_processed_path, index=False)
        train_original.to_excel(train_original_path, index=False)
        valid_original.to_excel(valid_original_path, index=False)

        fold_summary_rows.append({
            "Fold": fold_number,
            "Train_N": len(y_train),
            "Valid_N": len(y_valid),
            "Train_Class_0": int((y_train == 0).sum()),
            "Train_Class_1": int((y_train == 1).sum()),
            "Valid_Class_0": int((y_valid == 0).sum()),
            "Valid_Class_1": int((y_valid == 1).sum()),
            "Processed_Feature_Count": X_train_processed.shape[1],
            "Train_Processed_File": train_processed_path,
            "Valid_Processed_File": valid_processed_path,
        })

        print(f"Fold {fold_number} saved.")
        print(f"Train shape: {X_train_processed.shape}")
        print(f"Valid shape: {X_valid_processed.shape}")

    fold_summary = pd.DataFrame(fold_summary_rows)

    feature_group_rows = []

    for col in CONTINUOUS_NUMERIC_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "Continuous numeric",
            "Imputation": "Median",
            "Transformation": "StandardScaler",
        })

    for col in SKEWED_NUMERIC_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "Skewed numeric",
            "Imputation": "Median + missing indicator",
            "Transformation": "log1p + StandardScaler",
        })

    for col in PERCENT_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "Percent or index",
            "Imputation": "Median",
            "Transformation": "Divide by 100 + StandardScaler",
        })

    for col in BINARY_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "Binary",
            "Imputation": "Most frequent",
            "Transformation": "No scaling",
        })

    for col in CATEGORICAL_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "Multiclass categorical",
            "Imputation": "Most frequent",
            "Transformation": "One-hot encoding",
        })

    for col in HLA_COUNT_COLUMNS:
        feature_group_rows.append({
            "Column": col,
            "Group": "HLA count",
            "Imputation": "Median + missing indicator",
            "Transformation": "StandardScaler",
        })

    feature_groups = pd.DataFrame(feature_group_rows)

    summary_file = os.path.join(output_folder, "preprocessing_summary.xlsx")

    with pd.ExcelWriter(summary_file, engine="openpyxl") as writer:
        df_cleaned.to_excel(writer, sheet_name="Target_Cleaned_Data", index=False)
        removed_rows.to_excel(writer, sheet_name="Removed_Target_Rows", index=False)
        fold_summary.to_excel(writer, sheet_name="CV_Fold_Summary", index=False)
        feature_groups.to_excel(writer, sheet_name="Feature_Groups", index=False)
        pd.DataFrame({"Existing_Features": existing_features}).to_excel(
            writer,
            sheet_name="Existing_Features",
            index=False
        )
        pd.DataFrame({"Missing_Requested_Features": missing_features}).to_excel(
            writer,
            sheet_name="Missing_Features",
            index=False
        )

    print("\n============================================")
    print("CV preprocessing completed successfully.")
    print(f"Output folder: {output_folder}")
    print(f"Summary file: {summary_file}")
    print("============================================")

    return fold_summary