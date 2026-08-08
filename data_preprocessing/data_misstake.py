import pandas as pd
import numpy as np


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\number.xlsx"
output_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\number_cleaned_numeric.xlsx"


# ============================================================
# Read Excel file
# ============================================================

df = pd.read_excel(input_file, header=1)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# ID column
# ============================================================
id_col = "Rec_code"

if id_col not in df.columns:
    raise ValueError(
        f"Column '{id_col}' not found.\n"
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Columns that must be numeric
# ============================================================
numeric_columns = [col for col in df.columns if col != id_col]


# ============================================================
# Helper functions
# ============================================================
def is_real_empty(value):
    """
    True only for real empty/missing cells.
    Invalid symbols like '-', '–', '?', '#DIV/0!' are NOT considered real empty.
    """
    if pd.isna(value):
        return True

    value_str = str(value).strip()

    if value_str == "":
        return True

    return False


def clean_numeric_value(value):
    """
    Convert a cell value to numeric if possible.
    Invalid non-numeric values become NaN.

    Returns:
        cleaned_value, is_problem
    """

    # Real empty cells remain NaN but are not reported as problems
    if is_real_empty(value):
        return np.nan, False

    original_str = str(value).strip()

    # Remove invisible characters
    value_str = original_str.replace("\u200c", "")  # zero-width non-joiner
    value_str = value_str.replace("\u200f", "")    # right-to-left mark
    value_str = value_str.replace("\u200e", "")    # left-to-right mark
    value_str = value_str.replace("\xa0", "")      # non-breaking space
    value_str = value_str.strip()

    # Convert Persian/Arabic digits to English digits
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"

    for p, e in zip(persian_digits, english_digits):
        value_str = value_str.replace(p, e)

    for a, e in zip(arabic_digits, english_digits):
        value_str = value_str.replace(a, e)

    # Remove spaces inside number
    value_str = value_str.replace(" ", "")

    # Convert decimal comma to decimal dot
    value_str = value_str.replace(",", ".")

    # Common invalid symbols, texts, and Excel formula errors
    invalid_values = {
        "-", "–", "—", "−", "_",
        "?", "؟",
        ".", "..", "...",
        "nan", "NaN", "NAN",
        "none", "None",
        "null", "NULL",
        "خالی", "نامشخص", "نامعلوم",
        "ندارد", "مشخص نیست",

        # Excel formula errors
        "#DIV/0!",
        "#VALUE!",
        "#REF!",
        "#N/A",
        "#NAME?",
        "#NUM!",
        "#NULL!",
    }

    if value_str in invalid_values:
        return np.nan, True

    # Any other Excel-like error
    if value_str.startswith("#"):
        return np.nan, True

    # Fix obvious double-dot typo like 1..36
    if value_str.count(".") > 1:
        fixed_value = value_str.replace("..", ".")
        try:
            return float(fixed_value), True
        except ValueError:
            return np.nan, True

    # Try numeric conversion
    try:
        numeric_value = float(value_str)
        return numeric_value, False
    except ValueError:
        return np.nan, True


# ============================================================
# Clean data and create problems report
# ============================================================
cleaned_df = df.copy()
problems = []

for col in numeric_columns:

    cleaned_values = []

    for idx, value in cleaned_df[col].items():

        cleaned_value, is_problem = clean_numeric_value(value)
        cleaned_values.append(cleaned_value)

        if is_problem:
            problems.append({
                "Excel_Row": idx + 3,  # because header=1 and Python index starts from 0
                "Rec_code": cleaned_df.loc[idx, id_col],
                "Column_Name": col,
                "Original_Value": value,
                "Cleaned_Value": cleaned_value,
                "Action": "Converted to NaN or corrected"
            })

    cleaned_df[col] = cleaned_values


# ============================================================
# Problems report
# ============================================================
problems_df = pd.DataFrame(problems)

if problems_df.empty:
    problems_df = pd.DataFrame([{
        "Excel_Row": "",
        "Rec_code": "",
        "Column_Name": "",
        "Original_Value": "",
        "Cleaned_Value": "",
        "Action": "No problematic non-numeric values found"
    }])


# ============================================================
# Save output Excel file
# ============================================================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    cleaned_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    problems_df.to_excel(writer, sheet_name="Problems_Report", index=False)


# ============================================================
# Print summary
# ============================================================
print("Done!")
print(f"Cleaned file saved as: {output_file}")

print("\nNumber of problematic values found:")
print(len(problems))

print("\nProblem summary by column:")
if len(problems) > 0:
    print(problems_df["Column_Name"].value_counts())
else:
    print("No problems found.")