import pandas as pd
import numpy as np


# ============================================================
# File paths
# ============================================================
input_excel = r"D:\WORK\KD.xlsx"
output_excel = r"D:\WORK\KD_with_KDPI_KDRI.xlsx"

kdpi_code_file = r"D:\WORK\KDPI_KDRI.py"


# ============================================================
# Load only the original calculator core from KDPI_KDRI.py
# This prevents the example code at the bottom of KDPI_KDRI.py from running
# ============================================================
with open(kdpi_code_file, "r", encoding="utf-8") as f:
    code_text = f.read()

# Keep everything before the example section
# This assumes your example starts with "#Example"
if "#Example" in code_text:
    code_text = code_text.split("#Example")[0]

namespace = {}
exec(code_text, namespace)

KDRICalculator = namespace["KDRICalculator"]


# ============================================================
# Helper functions
# ============================================================
def to_float(value):
    """
    Convert Excel values to float.
    Returns NaN if conversion is not possible.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return np.nan


def to_bool(value):
    """
    Convert common Excel values to boolean.
    Missing values are treated as False.
    """
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    true_values = {
        "1", "1.0", "true", "yes", "y",
        "بله", "بلی", "مثبت", "دارد", "yes "
    }

    false_values = {
        "0", "0.0", "false", "no", "n",
        "خیر", "نه", "منفی", "ندارد", ""
    }

    if value in true_values:
        return True

    if value in false_values:
        return False

    # Any unknown value is conservatively treated as False
    return False


def calculate_for_row(row, calculator):
    """
    Calculate KDPI and KDRI for one donor row.
    """

    age = to_float(row["Donor_Age"])
    height_cm = to_float(row["Donor_height"])
    weight_kg = to_float(row["Donor_weight"])
    serum_creatinine = to_float(row["Donor_Cr"])

    hypertension = to_bool(row["DONOR_HTN"])
    diabetes = to_bool(row["DONOR_DM"])
    cause_of_death_cva = to_bool(row["CVA"])
    dcd = to_bool(row["DCD"])

    required_numeric_values = {
        "Donor_Age": age,
        "Donor_height": height_cm,
        "Donor_weight": weight_kg,
        "Donor_Cr": serum_creatinine
    }

    missing_required = [
        col for col, val in required_numeric_values.items()
        if pd.isna(val)
    ]

    if missing_required:
        return pd.Series({
            "KDPI": np.nan,
            "KDRI": np.nan
        })

    result = calculator.calculate(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        hypertension=hypertension,
        diabetes=diabetes,
        cause_of_death_cva=cause_of_death_cva,
        serum_creatinine=serum_creatinine,
        dcd=dcd
    )

    return pd.Series({
        "KDPI": result["KDPI"],
        "KDRI": result["KDRI_SCALED"]
    })


# ============================================================
# Read Excel file
# ============================================================
df = pd.read_excel(input_excel, header=1)

# If your real column names are in the second row, use this instead:
# df = pd.read_excel(input_excel, header=1)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Check required columns
# ============================================================
required_columns = [
    "Donor_Age",
    "Donor_height",
    "Donor_weight",
    "DONOR_HTN",
    "DONOR_DM",
    "CVA",
    "Donor_Cr",
    "DCD"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"The following required columns were not found: {missing_columns}\n"
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Calculate KDPI and KDRI for all rows
# ============================================================
calculator = KDRICalculator()

df[["KDPI", "KDRI"]] = df.apply(
    lambda row: calculate_for_row(row, calculator),
    axis=1
)


# ============================================================
# Save output Excel file
# ============================================================
df.to_excel(output_excel, index=False)


# ============================================================
# Print summary
# ============================================================
print("Done!")
print(f"Processed file saved as: {output_excel}")

print("\nKDPI summary:")
print(df["KDPI"].describe())

print("\nKDRI summary:")
print(df["KDRI"].describe())

print("\nMissing KDPI count:")
print(df["KDPI"].isna().sum())

print("\nKDPI value counts:")
print(df["KDPI"].value_counts(dropna=False).sort_index())