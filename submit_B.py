import os
import zipfile
import pandas as pd
from utils import load_config

def make_submission(cfg, test_df, threshold_value=0.5, loc_threshold=0.5):
    test_df['label'] = (test_df['prob'] >= threshold_value).astype(int)
    test_df['threshold'] = threshold_value
    test_df['loc_threshold'] = loc_threshold
    scores_csv = os.path.join(cfg["paths"]["out_dir"], "scores.csv")
    test_df[['image_id', 'prob', 'label', 'threshold', 'loc_threshold']].to_csv(scores_csv, index=False)
    print(f"Saved scores to {scores_csv}")

    submission_zip = os.path.join(cfg["paths"]["out_dir"], f"{cfg['team_name']}_localization_masks.zip")
    with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(cfg["paths"]["out_dir"]):
            if file.endswith('.npz'):
                zipf.write(os.path.join(cfg["paths"]["out_dir"], file), file)
    print(f"Created submission zip: {submission_zip}")
    return submission_zip

if __name__ == "__main__":
    cfg = load_config("config_B.yaml")
    test_csv = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_test_probs.csv")
    test_df = pd.read_csv(test_csv)
    make_submission(cfg, test_df)