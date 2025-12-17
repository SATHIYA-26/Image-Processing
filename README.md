# Image-Processing

🔹 Task A – Real vs Synthetic Image Detection
📝 Task Description

Given an image, predict whether it is:

0 → Real

1 → Synthetic

Two runs are required:

Constrained Run – Only official datasets

Open Run – Any additional external data allowed

📂 Task A Folder Structure
project_directory/
├── dataset/
│   ├── train/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── val/
│       ├── 0_real/
│       └── 1_fake/
│
├── dataset_open/              # Used only for Open Run
│   ├── train/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── val/
│       ├── 0_real/
│       └── 1_fake/
│
├── outputs/
├── submission/
│
├── train.py
├── infer.py
├── submit.py
├── main.py
├── model.py
├── utils.py
├── config_constrained.yaml
├── config_open.yaml

⚙️ Model

Backbone: ConvNext-B

Architecture: CombinedModel (classification branch only)

Training Strategy:

Step 1: Train classifier head

Step 2: Fine-tune last backbone layers

🚀 Task A Commands & Outputs
🔸 Training
python main.py --config config_constrained.yaml --action train


Outputs (outputs/):

<run_name>_classifier_model.pth

<run_name>_finetuned_model.pth

<run_name>_val_probs.csv

🔸 Inference
python main.py --config config_constrained.yaml --action infer_test \
--ckpt outputs/<run_name>_finetuned_model.pth \
--test_dir data/test


Output:

<run_name>_test_probs.csv


Format:

image_id,prob
img001.jpg,0.823
img002.jpg,0.124

🔸 Submission
python main.py --config config_constrained.yaml --action submit


Output:

submission/
└── teamname_constrained.zip


CSV inside ZIP:

image_id,prob,label,threshold
img001.jpg,0.823,1,0.5
img002.jpg,0.124,0,0.5


Repeat the same process with config_open.yaml for the Open Run.

🔹 Task B – Manipulated Region Localization
📝 Task Description

For each image:

Predict whether it is manipulated (classification)

Predict a pixel-level probability mask indicating manipulated regions

📂 Task B Dataset Structure
Training – TGIF Dataset
TGIF/
├── orig/
├── ps-sp/
├── sd2-sp/
├── sd2-fr/
├── sdxl-fr/
└── masks/

Validation – COCO + RAISE
validation/
├── coco/
│   ├── original/
│   ├── brushnet/
│   │   ├── image/
│   │   └── mask/
│   └── ...
└── raise/

⚙️ Model

Architecture: CombinedModel (classification + segmentation)

Loss:

Classification → CrossEntropy

Localization → BCE / Dice

Config:

mask_weight: 1.0

🚀 Task B Outputs
🔸 Mask Files

One .npz file per test image

Filename must match image name

fEdOddAW3EeT.npz


Contents:

(H, W) float16 array with values in [0.0, 1.0]

🔸 scores.csv
image_id,prob,label,threshold,loc_threshold
img001.jpg,0.715,1,0.5,0.5
img002.jpg,0.042,0,0.5,0.5

🔸 Submission ZIP
teamname_localization_masks.zip
├── scores.csv
├── img001.npz
├── img002.npz
└── ...
