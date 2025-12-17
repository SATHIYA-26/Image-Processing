import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from model_B import UNet
from utils import load_config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress tracking
from utils import load_config, seed_everything



# Global dataset class for training
class TaskBDataset:
    def __init__(self, root, transform=None, mask_transform=None, img_size=224):
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_size = img_size
        self.imgs = []
        if "train" in root:
            classes = ['orig', 'ps-sp', 'sd2-sp', 'sd2-fr', 'sdxl-fr']
            for class_name in classes:
                img_dir = os.path.join(root, class_name)
                if os.path.exists(img_dir):
                    mask_dir = os.path.join(root, class_name, 'mask') if class_name != 'orig' else None
                    for file in os.listdir(img_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(img_dir, file)
                            mask_file = f"{os.path.splitext(file)[0]}.png"
                            mask_path = os.path.join(mask_dir, mask_file) if mask_dir and os.path.exists(mask_dir) else None
                            self.imgs.append((img_path, mask_path, 1 if class_name != 'orig' else 0))
        elif "val" in root:
            for source in ['coco', 'raise']:
                base_dir = os.path.join(root, source)
                for method in ['brushnet', 'controlnet', 'hdpainter', 'inpaintanything', 'mixed', 'powerpaint', 'removeanything']:
                    img_dir = os.path.join(base_dir, method, 'image')
                    mask_dir = os.path.join(base_dir, method, 'mask')
                    if os.path.exists(img_dir):
                        for file in os.listdir(img_dir):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(img_dir, file)
                                mask_file = f"{os.path.splitext(file)[0]}.png"
                                mask_path = os.path.join(mask_dir, mask_file)
                                self.imgs.append((img_path, mask_path, 1))
                orig_dir = os.path.join(base_dir, 'original')
                if os.path.exists(orig_dir):
                    for file in os.listdir(orig_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(orig_dir, file)
                            self.imgs.append((img_path, None, 0))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, mask_path, target = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') if mask_path else Image.new('L', (self.img_size, self.img_size), 0)
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask, target

def train(cfg):
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]
    img_size = cfg["data"]["img_size"]
    batch_size = cfg["data"]["batch_size"]
    num_epochs = cfg["training"]["num_epochs"]
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"]["weight_decay"]
    mask_weight = cfg["training"]["mask_weight"]
    device = torch.device(cfg["device"])

    # Define transforms
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # Create datasets and loaders
    image_datasets = {
        "train": TaskBDataset(train_dir, data_transforms["train"], mask_transform),
        "val": TaskBDataset(val_dir, data_transforms["val"], mask_transform)
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"),
                      num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"))
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

    # Load model
    model = UNet(n_channels=3, n_classes=2).to(device)
    class_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # Training loop
    def train_model(model, class_criterion, mask_criterion, optimizer, scheduler, num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, masks, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                    inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)  # [batch, 3, H, W]
                        class_out = outputs[:, :2, :, :].mean([2, 3])  # [batch, 2]
                        mask_out = outputs[:, 2:, :, :]  # [batch, 1, H, W]
                        class_loss = class_criterion(class_out, labels)
                        mask_loss = mask_criterion(mask_out, masks)
                        loss = class_loss + mask_weight * mask_loss
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(class_out, 1)
                    running_corrects += torch.sum(preds == labels)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    val_df = pd.DataFrame({
                        "image_id": [os.path.basename(path) for path, _, _ in image_datasets["val"].imgs],
                        "prob": torch.softmax(outputs[:, :2, :, :].mean([2, 3])[:, 1].cpu().detach().numpy(), dim=1)[:, 1],
                        "mask": torch.sigmoid(outputs[:, 2:, :, :]).cpu().detach().numpy()
                    })
                    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
                    val_df.to_csv(os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_val_results.csv"), index=False)
                if phase == "val":
                    scheduler.step(epoch_acc)

        print(f"\nTraining completed in {time.time() - start_time:.0f}s. Best val Acc: {best_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    # Train the model
    model = train_model(model, class_criterion, mask_criterion, optimizer, scheduler, num_epochs)
    ckpt_path = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_finetuned_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    cfg = load_config("config_B.yaml")
    seed_everything(cfg["seed"])  # Sets the seed before training starts
    train(cfg)