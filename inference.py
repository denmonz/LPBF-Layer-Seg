import sys
import os
import time
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from scipy.ndimage import gaussian_filter

from configs import config
from train import get_unet_efficientnet

# ====================================================================
# 1. SLIDING WINDOW + TEST-TIME AUGMENTATION ENGINE
# ====================================================================
def create_gaussian_window(patch_size, sigma=64):
    """Creates a 2D Gaussian weight matrix to feather patch borders seamlessly."""
    window_1d = np.zeros(patch_size)
    window_1d[patch_size // 2] = 1.0
    window_2d = gaussian_filter(window_1d, sigma=sigma, mode='constant')
    window_2d = window_2d / np.max(window_2d)

    mask = np.ones((patch_size, patch_size))
    mask[0, :] *= 0;
    mask[-1, :] *= 0;
    mask[:, 0] *= 0;
    mask[:, -1] *= 0
    return window_2d * gaussian_filter(mask, sigma=3)


@torch.no_grad()
def predict_full_lpbf_layer(model, image, patch_size=512, stride=256, device='cuda'):
    """Segments any variable resolution LPBF layer utilizing Test-Time Augmentation."""
    model.eval()
    H, W = image.shape

    transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    prediction_canvas = np.zeros((H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)
    gaussian_patch = create_gaussian_window(patch_size)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_start, y_end = y, y + patch_size
            x_start, x_end = x, x + patch_size
            if y_end > H: y_start, y_end = H - patch_size, H
            if x_end > W: x_start, x_end = W - patch_size, W

            patch = image[y_start:y_end, x_start:x_end]

            # Generate TTA Tensors (Averages out glare anomalies and locks onto fine cracks/voids)
            patch_tensor_0 = transform(image=np.expand_dims(patch, axis=-1))['image'].unsqueeze(0).to(device)
            patch_tensor_h = torch.flip(patch_tensor_0, dims=[2])
            patch_tensor_v = torch.flip(patch_tensor_0, dims=[3])
            patch_tensor_hv = torch.flip(patch_tensor_0, dims=[2, 3])

            with torch.amp.autocast('cuda'):
                prob_0 = torch.sigmoid(model(patch_tensor_0)).squeeze().cpu().numpy()
                prob_h = torch.flip(torch.sigmoid(model(patch_tensor_h)), dims=[2]).squeeze().cpu().numpy()
                prob_v = torch.flip(torch.sigmoid(model(patch_tensor_v)), dims=[3]).squeeze().cpu().numpy()
                prob_hv = torch.flip(torch.sigmoid(model(patch_tensor_hv)), dims=[2, 3]).squeeze().cpu().numpy()

            probs = (prob_0 + prob_h + prob_v + prob_hv) / 4.0

            prediction_canvas[y_start:y_end, x_start:x_end] += probs * gaussian_patch
            weight_canvas[y_start:y_end, x_start:x_end] += gaussian_patch

    return prediction_canvas / (weight_canvas + 1e-8)


# ====================================================================
# 2. METROLOGY & NOISE CLEANING ENGINE
# ====================================================================
def clean_mask_anomalies(binary_mask, min_pixel_size=3):
    """Applies connected components to wipe out sub-resolution sensor noise grains."""
    mask_8u = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixel_size:
            cleaned_mask[labels == i] = 1

    return cleaned_mask


# ====================================================================
# 3. CONSOLE EXECUTION WRAPPER
# ====================================================================
def main():
    # ----------------------------------------------------
    # HARDCODED CALIBRATIONS (Match your 100-epoch sweet-spots)
    # ----------------------------------------------------
    WEIGHTS_PATH = Path(config.CHECKPOINT_FILE)
    OPTIMAL_THRESHOLD = config.OPTIMAL_THRESHOLD
    MIN_DEFECT_SIZE_PIXELS = config.MIN_DEFECT_PIXEL_SIZE

    # Check for drag-and-drop or terminal path arguments
    if len(sys.argv) < 2 and config.INFERENCE_DATA_PATH == "":
        print("Usage error: Please provide the absolute path to your target .tif image.")
        print('Example: python predict_single_layer.py "D:/Data/sample_layer.tif"')
        print("Or, replace INFERENCE_DATA_PATH in config.py")
        return

    input_image_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(config.INFERENCE_DATA_PATH)

    if not input_image_path.exists():
        print(f"Error: Target image file does not exist at: {input_image_path.resolve()}")
        return

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing fine-tuned network weights checkpoint at: {WEIGHTS_PATH.resolve()}")

    device = config.DEVICE

    # Import model schema definition from train.py
    model = get_unet_efficientnet(encoder_name="efficientnet-b3")

    # Restore model checkpoints
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load raw grayscale image
    raw_layer = cv2.imread(str(input_image_path), cv2.IMREAD_GRAYSCALE)
    if raw_layer is None:
        print(f"Error: OpenCV failed to parse image array format at: {input_image_path.name}")
        return

    print(f"Processing layer grid [{raw_layer.shape[0]}x{raw_layer.shape[1]}] using Tensor Cores...")
    start_time = time.time()

    # Execute sliding prediction canvas
    prob_map = predict_full_lpbf_layer(model, raw_layer, patch_size=512, stride=256, device=device)
    raw_binary_mask = (prob_map > OPTIMAL_THRESHOLD).astype(np.uint8)

    # Clean high-frequency noise grains
    cleaned_mask = clean_mask_anomalies(raw_binary_mask, min_pixel_size=MIN_DEFECT_SIZE_PIXELS)

    # Construct target destination path in the exact same directory with a .png extension
    output_png_path = input_image_path.parent / f"{input_image_path.stem}.png"

    # Write mask file multiplied by 255 so defects are crisp white shapes viewable in Windows
    cv2.imwrite(str(output_png_path), cleaned_mask * 255)

    print(f"Inference finalized in {time.time() - start_time:.2f} seconds.")
    print(f"Successfully generated predicted mask at: {output_png_path.resolve()}")


if __name__ == "__main__":
    main()