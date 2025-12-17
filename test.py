import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import load_config
from timm import create_model

def predict_image(img_path, model, device, transform):
    """Predict if the image is real or fake (Task A only)."""
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        class_out = model(image)  # Single output for Task A
        _, predicted = torch.max(class_out, 1)
    return "real" if predicted.item() == 0 else "fake"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--sample_image", required=True, help="Path to single image")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    device = torch.device(cfg["device"])

    # Transform
    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load model (match training architecture)
    model = create_model('convnext_base', pretrained=False, num_classes=cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # Predict
    result = predict_image(args.sample_image, model, device, transform)
    print(f"Prediction for {args.sample_image}: {result}")