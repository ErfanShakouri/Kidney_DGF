import pandas as pd
import numpy as np


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\LOS_DAYS.xlsx"
output_file = r"D:\WORK\LOS_DAYS_grouped.xlsx"


# ============================================================
# Read Excel file
# ============================================================
# If column names are in the second row, use header=1
df = pd.read_excel(input_file, header=1)

# If column names are in the first row, use this instead:
# df = pd.read_excel(input_file)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Target column
# ============================================================
los_col = "LOS_Days"

if los_col not in df.columns:
    raise ValueError(
        f"Column '{los_col}' not found. Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Convert LOS_Days to numeric
# Non-numeric values become NaN
# ============================================================
df[los_col] = pd.to_numeric(df[los_col], errors="coerce")


# ============================================================
# Create LOS group
# ============================================================
def classify_los(value):
    if pd.isna(value):
        return "Unknown"
    elif 0 <= value <= 13:
        return "Early"
    elif 14 <= value <= 21:
        return "Med"
    elif value > 21:
        return "Late"
    else:
        return "Unknown"


df["LOS_Group"] = df[los_col].apply(classify_los)


# ============================================================
# Optional: create summary table
# ============================================================
summary = (
    df["LOS_Group"]
    .value_counts(dropna=False)
    .reset_index()
)

summary.columns = ["LOS_Group", "Count"]
summary["Percent"] = summary["Count"] / summary["Count"].sum() * 100


# ============================================================
# Save output
# ============================================================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Data_with_LOS_Group", index=False)
    summary.to_excel(writer, sheet_name="LOS_Group_Summary", index=False)


print("Done!")
print(f"Output file saved as: {output_file}")
print("\nLOS Group Summary:")
print(summary)