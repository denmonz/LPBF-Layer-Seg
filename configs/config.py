import os
import torch

# Hardware Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True          # True allows fast asynchronous GPU memory transfers
NUM_WORKERS = 8            # Number of CPU cores assigned to fetch data (tweak based on CPU)

# Hyperparameters (Optimized for RTX 3090 24GB VRAM)
LEARNING_RATE = 3e-4 # Previously: 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
PATCH_SIZE = 512

# Dataset Directory Structures
"""
The following dataset structure is necessary to configure manually:
    Peregrine /                             <--- DATABASE_BASE_DIR
        Peregrine Dataset v2021-03 /
        Peregrine Dataset v2022-10.1 /
        Peregrine Dataset v2025-09 /
"""
BINARY_CLASSIFICATION = True # Binary ['no_defect', 'defect'] if True, or multi-class if False
DATASET_BASE_DIR = "D:/Data/Peregrine"
UNET_DATASET_DIR = f"{DATASET_BASE_DIR}/Unified_Unet_Dataset"

# File Paths
CHECKPOINT_DIR = "./models/UNet/checkpoints"
CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/best_unetplusplus_efficientnet-b3.pth"

# Inference Parameters
TEST_IMAGE_DIR = f"{UNET_DATASET_DIR}/test"
OUTPUT_DIR = f"{UNET_DATASET_DIR}/output"
ANALYSIS_REPORT = f"{UNET_DATASET_DIR}/analysis_report"
OPTIMAL_THRESHOLD = 0.50
MIN_DEFECT_PIXEL_SIZE = 3  # Noise suppression floor filter
PIXEL_SCALE_UM = 2.5  # Metrology conversion factor