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
from timm import create_model
from utils import load_config, seed_everything
from tqdm import tqdm

# ================== DATASET ==================
class SyntheticDataset:
    def __init__(self, root, transform=None, img_size=224):
        self.transform = transform
        self.img_size = img_size
        self.imgs = []
        self.classes = {'real': 0, 'fake': 1}

        print(f"Initializing dataset for root: {root}")

        # Handle train real images only from train/real/train2014
        if "train" in os.path.basename(root):
            coco_dir = os.path.join(root, "real", "train2014")
            if os.path.exists(coco_dir):
                for file in os.listdir(coco_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.imgs.append((os.path.join(coco_dir, file), self.classes['real']))
            else:
                print(f"Warning: train2014 directory not found at {coco_dir}")

            # Handle fake images from Fake1 to Fake7
            fake_dir = os.path.join(root, "fake")
            if os.path.exists(fake_dir):
                for subdir in os.listdir(fake_dir):
                    subdir_path = os.path.join(fake_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for file in os.listdir(subdir_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.imgs.append((os.path.join(subdir_path, file), self.classes['fake']))

        # Handle val images from ITW-SM/0_real and 1_fake
        elif "val" in os.path.basename(root):
            itw_sm_dir = os.path.join(root, "ITW-SM")
            print(f"Checking ITW-SM directory: {itw_sm_dir}")
            if os.path.exists(itw_sm_dir):
                for class_name in ['0_real', '1_fake']:
                    class_path = os.path.join(itw_sm_dir, class_name)
                    print(f"Checking class path: {class_path}")
                    if os.path.exists(class_path):
                        for file in os.listdir(class_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.imgs.append((os.path.join(class_path, file), self.classes[class_name.split('_')[1]]))
                    else:
                        print(f"Warning: {class_path} not found")
            else:
                print(f"Warning: ITW-SM directory not found in {root}")

        if not self.imgs:
            raise ValueError(f"No images found in {root}. Check directory structure and permissions.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            img_path, target = self.imgs[index]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None


# ================== TRAINING ==================
def train(cfg):
    device = torch.device(cfg["device"])
    img_size = cfg["data"]["img_size"]
    batch_size = cfg["data"]["batch_size"]
    num_epochs_step1 = cfg["training"]["num_epochs_step1"]
    num_epochs_step2 = cfg["training"]["num_epochs_step2"]
    lr_step1 = cfg["training"]["lr_step1"]
    lr_step2 = cfg["training"]["lr_step2"]
    weight_decay = cfg["training"]["weight_decay"]
    num_classes = cfg["model"]["num_classes"]

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

    image_datasets = {
        "train": SyntheticDataset(cfg["data"]["train_dir"], data_transforms["train"]),
        "val": SyntheticDataset(cfg["data"]["val_dir"], data_transforms["val"])
    }
    dataloaders = {
        x: DataLoader([item for item in image_datasets[x] if item[0] is not None],
                      batch_size=batch_size, shuffle=(x == "train"),
                      num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"))
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}
    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

    model = create_model('convnext_base', pretrained=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    def train_model(model, criterion, optimizer, num_epochs, phase_name):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"{phase_name} - Epoch {epoch+1}/{num_epochs}")
            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()
                running_loss = 0.0
                running_corrects = 0
                all_val_probs = []
                all_val_image_ids = []

                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    if inputs is None or labels is None:
                        continue
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        class_out = model(inputs)
                        _, preds = torch.max(class_out, 1)
                        loss = criterion(class_out, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    if phase == "val":
                        probs = torch.softmax(class_out, dim=1)[:, 1].cpu().numpy()
                        all_val_probs.extend(probs)
                        all_val_image_ids.extend([os.path.basename(path) for path, _ in image_datasets[phase].imgs[:len(labels)] if path])

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    val_df = pd.DataFrame({"image_id": all_val_image_ids, "prob": all_val_probs})
                    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
                    val_df.to_csv(os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_val_probs.csv"), index=False)

        print(f"{phase_name} done in {time.time() - start_time:.0f}s. Best val Acc: {best_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    # Step 1: Freeze backbone and train classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.head.parameters(), lr=lr_step1, weight_decay=weight_decay)
    model = train_model(model, criterion, optimizer, num_epochs_step1, "Step 1: Train Classifier")
    ckpt_path_step1 = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_classifier_model.pth")
    torch.save(model.state_dict(), ckpt_path_step1)

    # Step 2: Fine-tune last two stages
    for i, (name, param) in enumerate(model.named_parameters()):
        if "stages.3" in name or "stages.2" in name:
            param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_step2, weight_decay=weight_decay)
    model = train_model(model, criterion, optimizer, num_epochs_step2, "Step 2: Fine-Tune")
    ckpt_path_step2 = os.path.join(cfg["paths"]["out_dir"], f"{cfg['run_name']}_finetuned_model.pth")
    torch.save(model.state_dict(), ckpt_path_step2)

    return ckpt_path_step2


if __name__ == "__main__":
    cfg = load_config("config_constrained.yaml")
    seed_everything(cfg["seed"])
    train(cfg)
    print("Training completed successfully.")