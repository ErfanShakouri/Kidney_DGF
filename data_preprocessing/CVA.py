import pandas as pd
import numpy as np


# =========================
# File paths
# =========================
input_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\CVA.xlsx"
output_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\CVA_processed.xlsx"


# =========================
# Read Excel file
# =========================
# Use this line if column names are in the first row
df = pd.read_excel(input_file, header=1)

# Use this line instead if column names are in the second row
# df = pd.read_excel(input_file, header=1)


# =========================
# Clean column names
# =========================
df.columns = df.columns.astype(str).str.strip()


# =========================
# Check target column
# =========================
target_col = "Cause_of_Death"

if target_col not in df.columns:
    raise ValueError(
        f"Column '{target_col}' not found in the Excel file. "
        f"Available columns are: {df.columns.tolist()}"
    )


# =========================
# Create CVA column
# =========================
df["CVA"] = np.where(
    df[target_col].astype(str).str.strip() == "Cerebrovascular_Hemorrhage",
    1,
    0
)


# =========================
# Save output file
# =========================
df.to_excel(output_file, index=False)

print("Done!")
print(f"Processed file saved as: {output_file}")

print("\nCVA value counts:")
print(df["CVA"].value_counts(dropna=False))