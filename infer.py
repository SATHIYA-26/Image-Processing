import torch
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from timm import create_model
import torch.nn as nn
from utils import load_config, seed_everything

def infer_folder(ckpt_path, test_dir, cfg):
    device = torch.device(cfg["device"])
    num_classes = cfg["model"]["num_classes"]
    img_size = cfg["data"]["img_size"]

    # Load model
    model = create_model('convnext_base', pretrained=False, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Define transform
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Collect image paths
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    # Inference
    probs = []
    image_ids = []
    seed_everything(cfg["seed"])  # Ensure reproducibility
    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = data_transforms(img).unsqueeze(0).to(device)
                class_out = model(img)
                prob = torch.softmax(class_out, dim=1)[:, 1].cpu().numpy()[0]
                probs.append(prob)
                image_ids.append(os.path.basename(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame({"image_id": image_ids, "prob": probs})
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{cfg['run_name']}_test_probs.csv")
    df.to_csv(out_file, index=False)
    print(f"Inference completed. Results saved to {out_file}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = infer_folder(args.ckpt, args.test_dir, cfg)