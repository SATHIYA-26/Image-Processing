import os
import argparse
from utils import load_config
from infer import infer_folder
from submit import make_submission_task_a

def main():
    parser = argparse.ArgumentParser(description="MediaEval 2025 Task A Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--action", choices=["train", "infer_test", "submit"], required=True)
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument("--test_dir", type=str, help="Directory with test images")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.action == "train":
        from train import train
        ckpt_path = train(cfg)
        print(f"Training completed. Checkpoint saved at {ckpt_path}")
    elif args.action == "infer_test":
        if not args.ckpt or not args.test_dir:
            raise ValueError("ckpt and test_dir are required for infer_test")
        test_df = infer_folder(args.ckpt, args.test_dir, cfg)
        out_file = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_test_probs.csv")
        test_df.to_csv(out_file, index=False)
        print(f"Inference completed. Results saved to {out_file}")
    elif args.action == "submit":
        if not args.ckpt:
            raise ValueError("ckpt is required for submit")
        test_csv = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_test_probs.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test CSV not found at {test_csv}. Run infer_test first.")
        submission_zip = make_submission_task_a(test_csv, cfg["run_name"], cfg)
        print(f"Submission ready at {submission_zip}")

if __name__ == "__main__":
    main()