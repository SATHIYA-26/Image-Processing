# Image Processing – MediaEval 2025
## Synthetic Image Detection

This repository contains the complete implementation for the **MediaEval 2025 Synthetic Image Detection Challenge**, covering:

- **Task A:** Real vs. Synthetic Image Classification  
- **Task B:** Manipulated Region Localization  

Each task is handled **independently** with separate datasets, training pipelines, and submission formats.

---

## 🔹 Task A – Real vs. Synthetic Image Detection

### 📌 Task Description
Given an image, predict whether it is:

- **0 → Real**
- **1 → Synthetic**

Two runs are required:
- **Constrained Run** – Only official datasets allowed
- **Open Run** – External or additional datasets allowed

---



### ⚙️ Task A Model

- **Backbone:** ConvNeXt-B  
- **Architecture:** CombinedModel (classification branch only)

---

### 🧠 Training Strategy

1. **Step 1:** Train classifier head  
2. **Step 2:** Fine-tune last backbone layers  

---

### 🚀 Task A Commands

#### Training
```bash
python main.py --config config_constrained.yaml --action train
Inference
bash
Copy code
python main.py --config config_constrained.yaml --action infer_test \
--ckpt outputs/<run_name>_finetuned_model.pth \
--test_dir data/test
Submission
bash
Copy code
python main.py --config config_constrained.yaml --action submit
Repeat the same steps using config_open.yaml for the Open Run.

📤 Task A Outputs
Training Outputs (outputs/):

<run_name>_classifier_model.pth

<run_name>_finetuned_model.pth

<run_name>_val_probs.csv

Inference Output:

Copy code
<run_name>_test_probs.csv
Submission ZIPs (submission/):

Copy code
teamname_constrained.zip
teamname_open.zip
CSV Format:

Copy code
image_id,prob,label,threshold
image_001.jpg,0.873,1,0.5
image_002.jpg,0.142,0,0.5
🔹 Task B – Manipulated Region Localization
📌 Task Description
For each image:

Predict whether it is manipulated

Predict a pixel-level probability mask identifying manipulated regions

📂 Task B Dataset Structure
Copy code
dataset_taskB/
├── TGIF/                     # Training dataset
│   ├── orig/
│   ├── ps-sp/
│   ├── sd2-sp/
│   ├── sd2-fr/
│   ├── sdxl-fr/
│   └── masks/
│
└── validation/               # Validation dataset
    ├── coco/
    └── raise/
⚙️ Task B Model
Architecture: CombinedModel (classification + segmentation)

Loss Function:

Classification → CrossEntropy Loss

Localization → Mask loss (BCE / Dice)

Config Setting:

yaml
Copy code
mask_weight: 1.0
📤 Task B Outputs
Mask Files
One .npz file per test image

Filename must match image name

Stored as (H, W) float16 probability array

Example:

Copy code
image_001.npz
scores.csv
Copy code
image_id,prob,label,threshold,loc_threshold
image_001.jpg,0.715,1,0.5,0.5
Submission ZIP
Copy code
teamname_localization_masks.zip
├── scores.csv
├── image_001.npz
├── image_002.npz
└── ...
