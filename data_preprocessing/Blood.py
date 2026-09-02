import pandas as pd
import numpy as np


# ===========================================================
# File path
# ============================================================
input_file = r"D:\WORK\category.xlsx"
output_file = r"D:\WORK\blood_type_ABO_processed.xlsx"


# ============================================================
# Specify blood type columns here
# ============================================================
blood_type_columns = [
    "Rec_Blood_Type",
    # "Donor_Blood_Type",  # Add this only if needed
]


# ============================================================
# Read Excel file
# ============================================================
df = pd.read_excel(input_file, header=1)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Check missing columns
# ============================================================
missing_columns = [col for col in blood_type_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"The following columns were not found: {missing_columns}\n"
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Function to remove Rh sign and keep only ABO group
# ============================================================
def extract_abo(value):
    """
    Convert blood type values such as O+, O-, A+, AB+ to ABO only.
    Missing or unknown values remain NaN.
    """
    if pd.isna(value):
        return np.nan

    value_clean = str(value).strip().upper()

    # Remove spaces
    value_clean = value_clean.replace(" ", "")

    # Remove positive / negative signs
    value_clean = value_clean.replace("+", "")
    value_clean = value_clean.replace("-", "")

    # Keep only valid ABO groups
    valid_abo = {"O", "A", "B", "AB"}

    if value_clean in valid_abo:
        return value_clean

    return np.nan


# ============================================================
# Create new ABO-only columns
# ============================================================
for col in blood_type_columns:
    df[f"{col}_ABO"] = df[col].apply(extract_abo)


# ============================================================
# Save output Excel file
# ============================================================
df.to_excel(output_file, index=False)

print("Done!")
print(f"Processed file saved as: {output_file}")

print("\nValue counts after ABO extraction:")
for col in blood_type_columns:
    print(f"\n{col}_ABO:")
    print(df[f"{col}_ABO"].value_counts(dropna=False))