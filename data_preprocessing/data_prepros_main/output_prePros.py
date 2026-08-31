# output_prePros.py

import pandas as pd
import numpy as np


# ============================================================
# Helper function: clean column names
# ============================================================
def clean_column_names(df):
    """
    Clean column names by removing extra spaces and replacing internal spaces with underscore.
    """
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


# ============================================================
# Helper function: check missing or invalid target values
# ============================================================
def is_missing_value(value):
    """
    Detect missing or invalid values in the target column.
    """

    if pd.isna(value):
        return True

    value_str = str(value).strip()

    missing_values = {
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
        "None",
        "none",
        "NULL",
        "null",
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
        "مشخص نیست"
    }

    if value_str in missing_values:
        return True

    return False


# ============================================================
# Main function: remove rows with missing output / target
# ============================================================
def remove_missing_output_rows(
    input_file,
    output_file,
    target_col="Rec_DGF",
    id_col="Rec_code",
    header_row=0
):
    """
    Read Excel file, check target column, remove rows with missing target values,
    print removed rows based on Rec_code, and save cleaned file.

    Parameters
    ----------
    input_file : str
        Path to input Excel file.

    output_file : str
        Path to output Excel file.

    target_col : str
        Name of output / target column. Default is Rec_DGF.

    id_col : str
        Patient ID column. Default is Rec_code.

    header_row : int
        Header row index for pandas.
        Use 0 if column names are in first Excel row.
        Use 1 if column names are in second Excel row.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        DataFrame after removing rows with missing target.
    removed_rows : pandas.DataFrame
        Removed rows report.
    """

    # ============================================================
    # Read Excel
    # ============================================================
    df = pd.read_excel(input_file, header=header_row, dtype=object)

    # Clean column names
    df = clean_column_names(df)

    # ============================================================
    # Check required columns
    # ============================================================
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    if id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found.\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    # ============================================================
    # Detect missing target rows
    # ============================================================
    missing_mask = df[target_col].apply(is_missing_value)

    removed_rows = df.loc[missing_mask, [id_col, target_col]].copy()

    # Excel row number
    # If header_row=0: first data row is Excel row 2
    # If header_row=1: first data row is Excel row 3
    removed_rows["Excel_Row"] = removed_rows.index + header_row + 2

    # ============================================================
    # Print report
    # ============================================================
    print("\n============================================")
    print("Output preprocessing started")
    print("============================================")
    print(f"Target column: {target_col}")
    print(f"ID column: {id_col}")
    print(f"Total rows before cleaning: {len(df)}")
    print(f"Rows with missing target: {missing_mask.sum()}")
    print("============================================")

    if len(removed_rows) > 0:
        print("\nRows removed because target is missing:")
        print(
            removed_rows[[id_col, "Excel_Row", target_col]]
            .to_string(index=False)
        )
    else:
        print("\nNo rows removed. Target column has no missing values.")

    # ============================================================
    # Remove rows
    # ============================================================
    df_cleaned = df.loc[~missing_mask].copy()

    # Optional: convert target to numeric if it is 0/1
    df_cleaned[target_col] = pd.to_numeric(df_cleaned[target_col], errors="coerce")

    print("\n============================================")
    print(f"Total rows after cleaning: {len(df_cleaned)}")
    print(f"{target_col} distribution after cleaning:")
    print(df_cleaned[target_col].value_counts(dropna=False))
    print("============================================")

    # ============================================================
    # Save output Excel
    # ============================================================
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_cleaned.to_excel(
            writer,
            sheet_name="Data_Target_Cleaned",
            index=False
        )

        removed_rows.to_excel(
            writer,
            sheet_name="Removed_Rows",
            index=False
        )

    print("\nDone!")
    print(f"Cleaned file saved as:\n{output_file}")

    return df_cleaned, removed_rows