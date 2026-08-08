import pandas as pd
import numpy as np


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\Main_Information.xlsx"
output_file = r"D:\WORK\binary_columns_only.xlsx"


# ============================================================
# Specify binary columns here
# ============================================================
binary_columns = [
    "Preemptive",
    "Prev_KT_No",
    "Rec_Sex",
    "Donor_Sex",
    "Dialysis_Type",
    "Donor_Type",
    # Add more column names here
]


# ============================================================
# Read Excel file
# ============================================================
df = pd.read_excel(input_file, header=1)


df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Check missing columns
# ============================================================
missing_columns = [col for col in binary_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"The following columns were not found in the Excel file: {missing_columns}\n"
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Convert yes/no values to 1/0
# ============================================================
def convert_yes_no(value):
    """
    Convert yes/no values to 1/0.
    Missing or unknown values remain NaN.
    """
    if pd.isna(value):
        return np.nan

    value_clean = str(value).strip().lower()

    yes_values = {
        "yes", "y", "1", "1.0", "true",
        "male","m", "مرد",
        "بله", "بلی", "دارد", "مثبت",
        "hd", "deceased"
        
    }

    no_values = {
        "no", "n", "0", "0.0", "false",
        "female", "f", "زن",
        "خیر", "نه", "ندارد", "منفی",
        "pd", "living"       
        
    }

    if value_clean in yes_values:
        return 1

    if value_clean in no_values:
        return 0

    return np.nan


# ============================================================
# Create output dataframe with only original and converted columns
# ============================================================
output_df = pd.DataFrame()

for col in binary_columns:
    output_df[f"{col}_original"] = df[col]
    output_df[f"{col}_binary"] = df[col].apply(convert_yes_no)


# ============================================================
# Save output Excel file
# ============================================================
output_df.to_excel(output_file, index=False)

print("Done!")
print(f"Processed file saved as: {output_file}")

print("\nValue counts after conversion:")
for col in binary_columns:
    print(f"\n{col}:")
    print(output_df[f"{col}_binary"].value_counts(dropna=False))