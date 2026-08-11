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
# If real column names are in the second row, use header=1
df = pd.read_excel(input_file, header=1)

# If real column names are in the first row, use this instead:
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
# Create LOS group column
# ============================================================
def classify_los_week(value):
    if pd.isna(value):
        return np.nan

    elif 0 <= value <= 7:
        return "Week1"

    elif 8 <= value <= 14:
        return "Week2"

    elif 15 <= value <= 21:
        return "Week3"

    elif value > 21:
        return "Long"

    else:
        return np.nan


df["LOS_Week_Group"] = df[los_col].apply(classify_los_week)


# ============================================================
# Summary table
# ============================================================
summary = (
    df["LOS_Week_Group"]
    .value_counts(dropna=False)
    .reset_index()
)

summary.columns = ["LOS_Week_Group", "Count"]
summary["Percent"] = summary["Count"] / summary["Count"].sum() * 100


# ============================================================
# Save output
# ============================================================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Data_with_LOS_Week_Group", index=False)
    summary.to_excel(writer, sheet_name="LOS_Week_Group_Summary", index=False)


print("Done!")
print(f"Output file saved as: {output_file}")

print("\nLOS Week Group Summary:")
print(summary)