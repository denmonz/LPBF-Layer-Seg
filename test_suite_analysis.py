import os
import time
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs import config
from train import get_unet_efficientnet

def generate_layer_overlay_gallery(df_summary, images_dir, masks_dir, output_dir, model, device,
                                   optimal_threshold=0.50):
    """
    Identifies and renders side-by-side performance comparison maps for the absolute
    best, worst, and median performing layers based on F1-Dice scores.
    """
    # Filter out layers that don't have matching ground-truth arrays to prevent index crashes
    df_valid = df_summary[df_summary['Pixel_F1_Dice'] > 0.0].copy()
    if df_valid.empty:
        print("Warning: No valid layer overlays can be rendered (all F1 scores are 0.0).")
        return

    # 1. Automatically locate our structural performance benchmarks
    best_idx = df_valid['Pixel_F1_Dice'].idxmax()
    worst_idx = df_valid['Pixel_F1_Dice'].idxmin()
    # Find the layer closest to the median F1 performance floor
    median_val = df_valid['Pixel_F1_Dice'].median()
    median_idx = (df_valid['Pixel_F1_Dice'] - median_val).abs().idxmin()

    selected_layers = [
        {"type": "BEST Performing Layer", "row": df_valid.loc[best_idx]},
        {"type": "MEDIAN Performance Baseline", "row": df_valid.loc[median_idx]},
        {"type": "WORST Performing Layer", "row": df_valid.loc[worst_idx]}
    ]

    # Setup 3-row grid canvas
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    transform = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

    print("\nGenerating side-by-side diagnostic overlay gallery...")

    for row_idx, target in enumerate(selected_layers):
        filename = target["row"]["Layer_Filename"]
        f1_score = target["row"]["Pixel_F1_Dice"]
        layer_type = target["type"]

        # Load matrices
        raw_img = cv2.imread(str(Path(images_dir) / filename), cv2.IMREAD_GRAYSCALE)

        # Convert the stem to a string first so it appends smoothly to the file extension
        mask_filename = f"{Path(filename).stem}.png"
        gt_mask = cv2.imread(str(Path(masks_dir) / mask_filename), cv2.IMREAD_GRAYSCALE)

        if raw_img is None or gt_mask is None:
            continue

        # Extract predictions via sliding patches
        from test_suite_analysis import predict_full_lpbf_layer, extract_defect_metrology
        prob_map = predict_full_lpbf_layer(model, raw_img, patch_size=512, stride=256, device=device)
        raw_binary_mask = (prob_map > optimal_threshold).astype(np.uint8)

        # Filter noise to match reporting specifications
        cleaned_pred, _ = extract_defect_metrology(raw_binary_mask, filename, min_pixel_size=3)

        # Normalize binary definitions
        gt_bool = (gt_mask >= 2) & (gt_mask != 4) if np.max(gt_mask) > 1 else (gt_mask == 1)
        pred_bool = (cleaned_pred == 1)

        # 2. Map Confusion Matrix Classes into distinct color masks
        tp_mask = pred_bool & gt_bool  # Green: Correctly caught defects
        fp_mask = pred_bool & ~gt_bool  # Yellow: False alarms
        fn_mask = ~pred_bool & gt_bool  # Red: Missed defects

        # Convert grayscale raw image to RGB to paint colors onto it
        base_color = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        overlay = base_color.copy()

        # Apply specific neon highlights
        overlay[tp_mask] = [0, 255, 0]  # Neon Green
        overlay[fp_mask] = [255, 255, 0]  # Neon Yellow
        overlay[fn_mask] = [255, 0, 0]  # Neon Red

        # Alpha-blend the color highlights over the grayscale metallurgy background
        # 70% raw image transparency / 30% crisp error highlight transparency
        blended_map = cv2.addWeighted(base_color, 0.7, overlay, 0.3, 0)

        # 3. Plotting Grid Assignments
        # Left Panel: Raw input frame showing print track visibility
        axes[row_idx, 0].imshow(raw_img, cmap='gray')
        axes[row_idx, 0].set_title(f"{layer_type}\nFile: {filename}", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Right Panel: Color-coded segmentation accuracy overlay maps
        axes[row_idx, 1].imshow(blended_map)
        axes[row_idx, 1].set_title(f"Diagnostic Error Map (F1-Dice: {f1_score:.4f})\nGreen=TP | Yellow=FP | Red=FN",
                                   fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')

    plt.tight_layout()
    gallery_path = Path(output_dir) / "test_suite_layer_gallery.png"
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated dynamic visual audit gallery at: {gallery_path.resolve()}")

# ====================================================================
# SLIDING WINDOW INFERENCE & SMOOTHING ENGINE
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
    """Segments full-scale variable resolution LPBF slices utilizing Test-Time Augmentation."""
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

            # TTA Tensors
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
# METROLOGY AND NOISE FILTERING ENGINE
# ====================================================================
def extract_defect_metrology(binary_mask, image_name, min_pixel_size=3, pixel_resolution_um=2.5):
    """Calculates physical defect sizes and logs individual spatial coordinates."""
    mask_8u = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    defect_records = []
    valid_defect_count = 0

    for i in range(1, num_labels):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        if area_pixels < min_pixel_size:
            continue

        valid_defect_count += 1
        cleaned_mask[labels == i] = 1

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        physical_area_um2 = area_pixels * (pixel_resolution_um ** 2)
        max_length_um = max(w, h) * pixel_resolution_um

        defect_records.append({
            "Source_Image": image_name,
            "Defect_ID": valid_defect_count,
            "Center_X_Pixel": int(round(cx)),
            "Center_Y_Pixel": int(round(cy)),
            "Area_Pixels": area_pixels,
            "Area_um2": round(physical_area_um2, 2),
            "Max_Length_um": round(max_length_um, 2),
            "BB_Width_Px": w,
            "BB_Height_Px": h
        })

    return cleaned_mask, defect_records


# ====================================================================
# ANALYSIS COMPILATION & GRAPH RENDERING
# ====================================================================
def generate_summary_charts(df_summary, output_dir):
    """Generates analytical performance distribution charts for the test suite."""
    plt.figure(figsize=(12, 5))

    # Subplot 1: Distribution of F1 Scores across test builds
    plt.subplot(1, 2, 1)
    plt.hist(df_summary['Pixel_F1_Dice'], bins=10, color='tab:green', edgecolor='black', alpha=0.7)
    plt.axvline(df_summary['Pixel_F1_Dice'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df_summary['Pixel_F1_Dice'].mean():.3f}")
    plt.title("F1-Dice Score Distribution Over Test Set", fontsize=11, fontweight='bold')
    plt.xlabel("F1-Dice Score")
    plt.ylabel("Frequency (Layers)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()

    # Subplot 2: Precision vs Recall Scatter plot to identify outliers
    plt.subplot(1, 2, 2)
    plt.scatter(df_summary['Pixel_Recall'], df_summary['Pixel_Precision'], color='tab:blue', alpha=0.7,
                edgecolors='black')
    plt.title("Precision vs. Recall Scatter Matrix", fontsize=11, fontweight='bold')
    plt.xlabel("Recall (Defect Capture Rate)")
    plt.ylabel("Precision (False Positive Mitigation)")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    chart_path = Path(output_dir) / "test_suite_performance_curves.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Generated analytical performance distribution charts at: {chart_path.resolve()}")


# ====================================================================
# MASTER ANALYSIS SCRIPT EXECUTION ENTRY POINT
# ====================================================================
def main():
    device = config.DEVICE

    # ----------------------------------------------------
    # PATHS AND METROLOGY CALIBRATIONS (Adjust to match setup!)
    # ----------------------------------------------------
    WEIGHTS_PATH = Path(config.CHECKPOINT_FILE)
    TEST_DIR = Path(config.TEST_DATA_DIR)
    OUTPUT_DIR = Path(config.ANALYSIS_REPORT)
    OUTPUT_DIR.mkdir(exist_ok=True)

    OPTIMAL_THRESHOLD = config.OPTIMAL_THRESHOLD
    MIN_DEFECT_SIZE_PIXELS = config.MIN_DEFECT_PIXEL_SIZE  # Noise suppression floor filter
    PIXEL_SCALE_UM = config.PIXEL_SCALE_UM  # Metrology conversion factor
    PATCH_SIZE = config.PATCH_SIZE

    # Initialize architecture & restore checkpoints
    model = get_unet_efficientnet(encoder_name="efficientnet-b3")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights file not found at: {WEIGHTS_PATH.resolve()}")

    print(f"Restoring fine-tuned model weights from: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    images_dir = TEST_DIR / "images"
    masks_dir = TEST_DIR / "masks"

    test_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    total_files = len(test_files)

    if total_files == 0:
        raise FileNotFoundError(f"No test images discovered inside: {images_dir}")

    print(f"\n--- Launching Complete Test Evaluation Bench over {total_files} full-scale layers ---")

    # Master accumulators for summary metrics and analytical performance tracking
    master_defect_log = []
    summary_records = []
    global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0
    start_time = time.time()

    for idx, filename in enumerate(test_files, 1):
        base_name = Path(filename).stem
        img_path = images_dir / filename
        mask_path = masks_dir / f"{base_name}.png"

        raw_layer = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw_layer is None:
            continue

        prob_map = predict_full_lpbf_layer(model, raw_layer, patch_size=PATCH_SIZE, stride=256, device=device)

        raw_binary_mask = (prob_map > OPTIMAL_THRESHOLD).astype(np.uint8)

        # Filter noise and extract spatial defect indices
        cleaned_mask, layer_defects = extract_defect_metrology(
            binary_mask=raw_binary_mask,
            image_name=filename,
            min_pixel_size=MIN_DEFECT_SIZE_PIXELS,
            pixel_resolution_um=PIXEL_SCALE_UM
        )
        master_defect_log.extend(layer_defects)

        # Pixel-wise overlap verification checking
        precision, recall, f1_score = 0.0, 0.0, 0.0
        gt_defect_count = 0

        if mask_path.exists():
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None and gt_mask.shape == raw_layer.shape:
                pred_bool = (cleaned_mask == 1)

                # ----------------------------------------------------
                # DYNAMIC MULTI-FORMAT MASKS PARSER
                # ----------------------------------------------------
                unique_labels = np.unique(gt_mask)

                # Condition A: Multi-Class Metadata Structure
                # (Triggered if the mask contains indices greater than 1, e.g., class 2 to 16)
                if np.max(unique_labels) > 1:
                    # In your multi-class system, classes >= 2 are defects
                    gt_bool = (gt_mask >= 2)

                # Condition B: Pure Binary Metadata Structure
                # (0 is background, 1 is defect)
                else:
                    # Explicitly treat pixels containing intensity 1 as the target defect area
                    gt_bool = (gt_mask == 1)
                # ----------------------------------------------------

                # Convert boolean mask to 8-bit array for OpenCV's connected components engine
                gt_8u = (gt_bool * 255).astype(np.uint8)
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(gt_8u, connectivity=8)
                gt_defect_count = max(0, num_labels - 1)

                # Run metric intersection mapping matrix calculations
                tp = np.sum(pred_bool & gt_bool)
                fp = np.sum(pred_bool & ~gt_bool)
                fn = np.sum(~pred_bool & gt_bool)
                tn = np.sum(~pred_bool & ~gt_bool)

                global_tp += tp;
                global_fp += fp;
                global_fn += fn;
                global_tn += tn

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        summary_records.append({
            "Layer_Filename": filename,
            "Resolution": f"{raw_layer.shape}x{raw_layer.shape}",
            "GT_Defect_Count": gt_defect_count,  # <-- NEW EXCEL OUTPUT COLUMN
            "Predicted_Defect_Count": len(layer_defects),  # Renamed for clarity vs Ground Truth
            "Total_Area_um2": round(sum([r["Area_um2"] for r in layer_defects]), 2),
            "Pixel_Precision": round(precision, 4),
            "Pixel_Recall": round(recall, 4),
            "Pixel_F1_Dice": round(f1_score, 4)
        })

        print(
            f"Processed [{idx}/{total_files}]: {filename} | Threshold: {config.OPTIMAL_THRESHOLD} | GT Count: {gt_defect_count} | Pred Count: {len(layer_defects)} | F1-Dice: {f1_score:.4f}")

    # Compile data summaries into pandas dataframes
    df_summary = pd.DataFrame(summary_records)

    # Calculate dataset-wide total analytics summary
    overall_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    overall_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    overall_f1 = 2 * global_tp / (2 * global_tp + global_fp + global_fn) if (2 * global_tp + global_fp + global_fn) > 0 else 0.0

    # ----------------------------------------------------
    # MACRO SUMMARY TRACKING ENGINE
    # ----------------------------------------------------
    # Filter out layers that are completely empty or have minor sensor noise (<= 5 defects)
    # This prevents division-by-zero or massive penalties from nearly blank layers
    df_active_layers = df_summary[df_summary['GT_Defect_Count'] > 0]

    if not df_active_layers.empty:
        macro_precision = df_active_layers['Pixel_Precision'].mean()
        macro_recall = df_active_layers['Pixel_Recall'].mean()
        macro_f1 = df_active_layers['Pixel_F1_Dice'].mean()
        active_layer_count = len(df_active_layers)
    else:
        macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
        active_layer_count = 0

    # ----------------------------------------------------
    # REPORT EXPORT GENERATION
    # ----------------------------------------------------
    df_summary.to_excel(OUTPUT_DIR / "test_suite_layer_summary.xlsx", index=False)
    if master_defect_log:
        pd.DataFrame(master_defect_log).to_excel(OUTPUT_DIR / "test_suite_itemized_defects_log.xlsx", index=False)

    # Render diagnostics graphics dashboards
    generate_summary_charts(df_summary, OUTPUT_DIR)

    # --- NEW ADDITION: LIVE ERROR INSIGHT OVERLAY ENGINE ---
    generate_layer_overlay_gallery(
        df_summary=df_summary,
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=OUTPUT_DIR,
        model=model,
        device=device,
        optimal_threshold=OPTIMAL_THRESHOLD
    )

    print("\n" + "="*60)
    print("         COMPLETE CRITICAL MODEL METRICS SUMMARY        ")
    print("="*60)
    print(f"Total Evaluated Test Layers:              {total_files}")
    print(f"Total Defects Localized Across Test Set:  {df_summary['Predicted_Defect_Count'].sum()}")
    print(f"Average Detected Defects Per Slice Layer: {df_summary['Predicted_Defect_Count'].mean():.2f}")
    print("-"*60)
    print(" >>> GLOBAL MICRO-AVERAGED METRICS (Pixel Volume Base)")
    print(f"  Dataset Micro Precision:                {overall_precision:.4f}")
    print(f"  Dataset Micro Recall:                   {overall_recall:.4f}")
    print(f"  Dataset Micro F1-Dice:                  {overall_f1:.4f}")
    print("-"*60)
    print(f" >>> ACTIVE MACRO-AVERAGED METRICS ({active_layer_count}/{total_files} Significant Layers)")
    print(f"  Dataset Macro Precision:                {macro_precision:.4f}")
    print(f"  Dataset Macro Recall:                   {macro_recall:.4f}")
    print(f"  Dataset Macro F1-Dice:                  {macro_f1:.4f}")
    print("-"*60)
    print(f"Worst Performing Layer (Lowest F1):       {df_summary.loc[df_summary['Pixel_F1_Dice'].idxmin()]['Layer_Filename']} (F1: {df_summary['Pixel_F1_Dice'].min():.4f})")
    print(f"Best Performing Layer (Highest F1):       {df_summary.loc[df_summary['Pixel_F1_Dice'].idxmax()]['Layer_Filename']} (F1: {df_summary['Pixel_F1_Dice'].max():.4f})")
    print(f"Total Core Processing Runtime Execution:  {time.time() - start_time:.2f} seconds")
    print("="*60)
    print(f"All metric sheets and charts exported successfully to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()