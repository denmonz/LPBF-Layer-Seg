# PBF-Layer-Seg
**Laser Powder Bed Fusion Layer Instance Segmentation for Anomaly / Defect Identification**

## Requirements
- [Python v3.12.10](https://www.python.org/downloads/release/python-31210/)
- [uv Python Package Manager](https://docs.astral.sh/uv/)
- [CUDA Toolkit v12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

## My System
The following is a break-down of my system configuration:
- CPU: AMD Ryzen 7 5800X 8-Core Processor
- GPU: NVIDIA GeForce RTX 3090 w/ 24GB VRAM
- RAM: 128GB DDR4

## Setup Instructions
### uv Package Manager
1. Open a command prompt in the root directory, and run `uv sync`. This will configure the necessary Python packages to run the code.

### Dataset Setup
1. Download the three Peregrine datasets:
- [v2021-03](https://doi.ccs.ornl.gov/dataset/e2decf63-021c-563c-8729-ffe02769176c)
- [v2022-10.1](https://doi.ccs.ornl.gov/dataset/a2bb0d19-d5ed-5964-bfee-f8ec21d4b912)
- [v2025-09](https://doi.ccs.ornl.gov/dataset/96b7da99-07e1-562d-88d5-d7e079acfef7)

2. Move the data into a parent `Peregrine` folder:
```
Peregrine/                          <--- DATABASE_BASE_DIR
├── Peregrine Dataset v2021-03/
├── Peregrine Dataset v2022-10.1/
├── Peregrine Dataset v2025-09/
└── Unified_Unet_Dataset/           <--- Automatically created
```

3. Download the model weights from [here], and place in `models/UNet/checkpoints/`.

### Configuration Setup
Configure the parameters of the repository, found within the `configs/config.py` file, such as:
1. NUM_WORKERS: Should be set to the number of cores available on your CPU.
2. DATASET_BASE_DIR: The full path to the root `Peregrine` folder, set in the previous section.

### Dataset Standardization and Train/Val/Test Split
With the above configured, run the following command to standardize and combine the three datasets into a `Unified_Unet_Dataset` folder:

`uv run python setup.py`

## Functionality
With the above configured, you can perform the following functionalities.

### Training
To train your own model on the data, run the following: 

`uv run python train.py`

### Testing
To test on the held-out testing data using the trained model, run the following: 

`uv run python test_suite_analysis.py`

### Inference
To conduct inference on your own data using the trained model, run the following: 

`uv run python inference.py`

## Overview and Analysis Files
For more metadata information, review the following write-ups:
- [Dataset Overview](data/Dataset_Overview.md)
- [Model Overview](models/UNet/Model_Overview.md)
- [Fine-Tuned Model Evaluation](models/UNet/checkpoints/Best_100_epochs/Model_Evaluation.md)

## TODO Checklist
In future iterations, I plan to develop the following capabilities and features:
- [ ] Automatically export trained weights to the following to speed-up inference:
  - [ ] ONNX
  - [ ] TensorRT
- [ ] Compare performance metrics against larger pre-trained EfficientNet Encoders
- [ ] Quantify epistemic uncertainty (model confidence) by integrating Mote Carlo (MC) dropout as a toggle for testing and inference