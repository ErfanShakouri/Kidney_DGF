# main_preprocess.py
from output_prePros import remove_missing_output_rows
from preprocess_cross_validation import run_cv_preprocessing

# ============================================================
# File paths
# ============================================================
input_file = r"D:\first_DGF_target_cleaned.xlsx"

output_folder = r"D:\DGF_Logistic_Preprocessed_CV"


# ============================================================
# Run preprocessing
# ============================================================

fold_summary = run_cv_preprocessing(
    input_file=input_file,
    output_folder=output_folder,
    target_col="Rec_DGF",
    id_col="Rec_code",
    header_row=0,
    n_splits=5,
    random_state=42
)
'''
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
'''