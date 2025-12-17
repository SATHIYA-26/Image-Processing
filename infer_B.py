import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from model_B import UNet
from utils import load_config
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress tracking
from utils import load_config, seed_everything

# Global dataset class for inference
class TestDataset:
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

def infer(cfg, ckpt_path, test_dir):
    device = torch.device(cfg["device"])
    img_size = cfg["data"]["img_size"]
    batch_size = cfg["data"]["batch_size"]

    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    test_dataset = TestDataset(test_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"))
    print(f"Loaded {len(test_dataset)} test images from {test_dir}")

    # Load model
    model = UNet(n_channels=3, n_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Inference
    all_probs = []
    all_masks = []
    all_image_ids = []
    with torch.no_grad():
        for inputs, image_ids in tqdm(test_loader, desc="Inferring"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            class_out = outputs[:, :2, :, :].mean([2, 3])  # [batch, 2]
            mask_out = outputs[:, 2:, :, :]  # [batch, 1, H, W]
            probs = torch.softmax(class_out, dim=1)[:, 1].cpu().numpy()
            mask_probs = torch.sigmoid(mask_out).cpu().numpy()
            all_probs.extend(probs)
            all_masks.extend(mask_probs)
            all_image_ids.extend(image_ids)

    # Save masks as .npz files
    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    for img_id, mask_prob in zip(all_image_ids, all_masks):
        np.savez_compressed(os.path.join(cfg["paths"]["out_dir"], f"{img_id}.npz"), mask=mask_prob.astype(np.float16))
    print(f"Saved {len(all_image_ids)} mask files to {cfg['paths']['out_dir']}")

    # Save probabilities for scores.csv
    test_df = pd.DataFrame({"image_id": all_image_ids, "prob": all_probs})
    test_df.to_csv(os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_test_probs.csv"), index=False)
    print(f"Saved probabilities to {cfg['run_name']}_test_probs.csv")
    return test_df

if __name__ == "__main__":
    cfg = load_config("config_B.yaml")
    seed_everything(cfg["seed"])  # Sets the seed before inference starts
    ckpt_path = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_finetuned_model.pth")
    infer(cfg, ckpt_path, os.path.join(cfg["data"]["test_dir"], "images"))