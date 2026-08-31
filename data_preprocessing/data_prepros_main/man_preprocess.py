# man_preprocess.py

from output_prePros import remove_missing_output_rows


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\first.xlsx"

output_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\first_DGF_target_cleaned.xlsx"


# ============================================================
# Run output preprocessing
# ============================================================
df_cleaned, removed_rows = remove_missing_output_rows(
    input_file=input_file,
    output_file=output_file,
    target_col="Rec_DGF",
    id_col="Rec_code",
    header_row=1
)

print("\nOutput preprocessing finished successfully.")