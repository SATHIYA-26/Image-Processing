import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(val_probs_path, val_labels_path=None):
    val_df = pd.read_csv(val_probs_path)
    if val_labels_path:
        try:
            val_labels = pd.read_csv(val_labels_path).set_index('image_id').reindex(val_df['image_id'])['label'].values
            if val_labels is None or len(val_labels) != len(val_df) or np.any(pd.isna(val_labels)):
                raise ValueError("Invalid or mismatched labels")
        except Exception as e:
            print(f"Error loading val_labels: {e}, cannot optimize threshold")
            return None
    else:
        print("Warning: No val_labels_path provided, cannot optimize threshold without labels")
        return None

    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (val_df['prob'] > thresh).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    print(f"Optimized F1: {best_f1:.4f} at threshold: {best_threshold}")
    return best_threshold

if __name__ == "__main__":
    val_probs_path = "D:/MediaEval_2025/data/outputs/efficientnet_b0_large_dataset_val_probs.csv"
    threshold = find_best_threshold(val_probs_path, val_labels_path="D:/MediaEval_2025/data/val_labels.csv")
    print(f"Best threshold: {threshold}")