#test with our samplings

import numpy as np
import pandas as pd
import glob
import os


# ==========================================================
# FUNCTION 1: Process Predictions & Create Comparison Columns
# ==========================================================
def compare_predictions(
    df, predictions, target_col="ground_truth", threshold=0.5
):
    """Processes predictor probabilities, applies threshold logic (> 0.5 for class 1),

    and adds 'prob_1', 'predicted_val', and 'Match' columns to the dataset.
    """
    result_df = df.copy()

    # Convert predictions to NumPy array if list/DataFrame
    preds_arr = np.array(predictions)

    # Extract class 1 probabilities (handles both 2D [[prob_0, prob_1]] and 1D [prob_1] formats)
    if preds_arr.ndim == 2 and preds_arr.shape[1] == 2:
        result_df["prob_1"] = preds_arr[:, 1]
    else:
        result_df["prob_1"] = preds_arr.flatten()

    # Threshold Logic: Keep 1 if prob_1 > threshold (0.5), otherwise 0
    result_df["predicted_val"] = (
        result_df["prob_1"] > threshold
    ).astype(int)

    # Check for match against Ground_Truth (1 = Match, 0 = Mismatch)
    result_df["Match"] = (
        result_df["predicted_val"] == result_df[target_col]
    ).astype(int)

    return result_df


# ==========================================================
# FUNCTION 2: Performance & Error Analysis
# ==========================================================
def analyze_predictions(
    df, target_col="ground_truth", pred_col="predicted_val"
):
    """Calculates accuracy metrics, match counts, and confusion matrix breakdown."""
    total = len(df)
    matches = (df[pred_col] == df[target_col]).sum()
    mismatches = total - matches
    accuracy = (matches / total) * 100 if total > 0 else 0

    # Classification breakdown
    tp = ((df[pred_col] == 1) & (df[target_col] == 1)).sum()  # True Positives
    tn = ((df[pred_col] == 0) & (df[target_col] == 0)).sum()  # True Negatives
    fp = ((df[pred_col] == 1) & (df[target_col] == 0)).sum()  # False Positives
    fn = ((df[pred_col] == 0) & (df[target_col] == 1)).sum()  # False Negatives

    analysis_summary = {
        "Total Records": total,
        "Correct Matches": matches,
        "Mismatches": mismatches,
        "Accuracy (%)": round(accuracy, 2),
        "True Positives (Actual 1, Pred 1)": tp,
        "True Negatives (Actual 0, Pred 0)": tn,
        "False Positives (Actual 0, Pred 1)": fp,
        "False Negatives (Actual 1, Pred 0)": fn,
    }

    print("========================================")
    print("      PREDICTION ANALYSIS REPORT        ")
    print("========================================")
    for metric, val in analysis_summary.items():
        print(f"{metric:<35}: {val}")
    print("========================================")

    return analysis_summary




def process_and_evaluate_all_csvs(
    predictor,
    folder_path="Sampling_data_to_test",
    target_col="ground_truth",
    drop_cols=["Driver"],
):
    """Reads all CSV files from a folder, runs predictor, performs per-file analysis,

    and outputs a cumulative analysis across all datasets combined.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{folder_path}'")
        return {}, None

    file_results = {}
    all_results_list = []

    for file_path in sorted(csv_files):
        filename = os.path.basename(file_path)
        print(f"\n----------------------------------------")
        print(f" Processing: {filename}")
        print(f"----------------------------------------")

        # 1. Read raw CSV
        df_raw = pd.read_csv(file_path)
        print(df_raw["ground_truth"].value_counts())
        # 2. Build feature matrix for model (drop non-feature columns)
        # Handles column dropping properly and ignores errors if column isn't present
        columns_to_remove = set(
            drop_cols + [target_col, target_col.lower(), "ground_truth"]
        )
        df_to_test = df_raw.drop(
            columns=[c for c in columns_to_remove if c in df_raw.columns]
        )
        

        # 3. Get predictions on features only
        df_out = predictor.predict_proba(df_to_test)
        
        # 4. Compare predictions against original Ground_Truth
        result_df = compare_predictions(
            df=df_raw, predictions=df_out, target_col=target_col
        )
        result_df["Source_File"] = filename  # Track source file

        # 5. Single-file analysis
        print(f"--- Per-File Analysis [{filename}] ---")
        analysis_summary = analyze_predictions(
            df=result_df, target_col=target_col, pred_col="predicted_val"
        )

        # Store outputs
        file_results[filename] = {
            "result_df": result_df,
            "analysis": analysis_summary,
        }
        all_results_list.append(result_df)

    # ==========================================================
    # CUMULATIVE ANALYSIS ACROSS ALL FILES
    # ==========================================================
    combined_results_df = pd.concat(all_results_list, ignore_index=True)

    print("\n" + "=" * 50)
    print("      OVERALL CUMULATIVE ANALYSIS REPORT        ")
    print("=" * 50)
    cumulative_summary = analyze_predictions(
        df=combined_results_df, target_col=target_col, pred_col="predicted_val"
    )

    return file_results, combined_results_df

