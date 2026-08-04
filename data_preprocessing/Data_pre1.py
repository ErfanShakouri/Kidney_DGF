import pandas as pd
import numpy as np

# Input Excel file path
input_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\Main_Information.xlsx"

# Output Excel file path
output_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\Main_Information_processed.xlsx"

# Read the Excel file
# header=1 means: use the second row of Excel as column names
df = pd.read_excel(input_file, header=1)

# Remove extra spaces from column names
df.columns = df.columns.astype(str).str.strip()

# Optional: print columns to check
print("Columns found in the Excel file:")
print(df.columns.tolist())

# Define the target column name
target_col = "Rejection/Expire"

# Check if the target column exists
if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in the Excel file.")

# Keep the original column for checking missing values
original_values = df[target_col]

# Standardize values: convert to string, remove extra spaces, and lowercase
values = df[target_col].astype(str).str.strip().str.lower()

# Create the Rejection column
# no      -> 0
# yes     -> 1
# expire  -> "null"
df["Rejection"] = values.map({
    "no": 0,
    "yes": 1,
    "expire": "null"
})

# Create the Expire column
# expire -> 1
# non-empty values -> 0
# empty values -> NaN
df["Expire"] = np.where(
    original_values.isna(),
    np.nan,
    np.where(values == "expire", 1, 0)
)

# Save the processed DataFrame to a new Excel file
df.to_excel(output_file, index=False)

print("Done!")
print(f"Processed file saved as: {output_file}")