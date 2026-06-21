import os
import sys
import h5py
import numpy as np
import tifffile as tiff
from collections import Counter
from PIL import Image
from scipy.ndimage import label
from configs import config
from data.peregrine2025_metadata import binary_class_dict, class_dict_2025, standardized_class_dict_2025, non_represented_classes

def count_distinct_anomaly_instances(hdf5_file_paths):
    """
    Groups adjacent pixels across 3D layers into distinct anomaly objects.
    Counts the total separate objects per class, per file, and combined.
    """
    global_instance_counts = Counter()
    file_summaries = {}

    # Define a 3D structural element (connectivity-26: includes diagonals)
    # This ensures anomalies touching across layers or corners are one object
    structure_3d = np.ones((3, 3, 3), dtype=np.uint8)

    for file_path in hdf5_file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        file_name = os.path.basename(file_path)
        file_counts = Counter()

        with h5py.File(file_path, 'r') as f:
            if 'slices/segmentation_results' in f:
                class_names = f['slices/segmentation_results'].attrs['class_names'].split(',')
                               # for c in f['slices/segmentation_results'].attrs['class_names'][:]]
            else:
                print(f"Skipping {file_name}: No class names found.")
                continue

            print(f"\nGrouping pixels into distinct objects for: {file_name}...")

            for class_id, class_name in enumerate(class_names):
                class_key = f"slices/segmentation_results/{class_id}"

                if class_key in f:
                    # Load the 3D binary volume [layers, height, width]
                    binary_volume = f[class_key][...]

                    # Group true pixels (==1) into distinct continuous objects
                    # labeled_array: matrix where each unique object gets its own integer ID (1, 2, 3...)
                    # num_features: the total number of unique objects found
                    labeled_array, num_features = label(binary_volume == 1, structure=structure_3d)

                    file_counts[class_name] = num_features
                    global_instance_counts[class_name] += num_features
                else:
                    file_counts[class_name] = 0

        file_summaries[file_name] = file_counts

        # Print per-file instance report
        print(f"{'-' * 40}\nObject Instance Count for {file_name}:")
        for class_name, count in file_counts.items():
            print(f"  • {class_name:<25} : {count:,} distinct objects")

    # Combined dataset summary report
    print(f"\n{'=' * 50}\nCOMBINED INSTANCE SUMMARY (All Files):\n{'=' * 50}")
    for class_name, total_instances in global_instance_counts.items():
        print(f"  • {class_name:<25} : {total_instances:,} total separate objects")

    return file_summaries, global_instance_counts

def setup_directories(base_path):
    """Creates the standard U-Net dataset structure."""
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'masks']
    for split in splits:
        for subdir in subdirs:
            os.makedirs(os.path.join(base_path, split, subdir), exist_ok=True)


def find_camera_dataset_path(f, desired_cams=None):
    """
    Dynamically walks down slices/camera_data to find the target visible
    light camera data path and avoid hardcoding name strings.
    """
    if 'slices/camera_data' not in f:
        return None

    cam_group = f['slices/camera_data']

    # 1. Look for any camera subfolders (e.g., 'visible', 'visible_light', or 'Basler_...')
    for cam_name in cam_group.keys():
        if desired_cams and cam_name not in desired_cams:
            continue
        sub_group_path = f"slices/camera_data/{cam_name}"
        sub_group = f[sub_group_path]

        # 2. Look for the frame ID (e.g., '0' for post-melt darkfield/brightfield)
        # We prefer '0' as per the documentation guide
        if '0' in sub_group:
            target_path = f"{sub_group_path}/0"
            print(f"--> Dynamically auto-detected camera dataset path: '{target_path}'")
            return target_path

        # Fallback: take the first key available if '0' isn't named explicitly
        if len(sub_group.keys()) > 0:
            first_frame_key = list(sub_group.keys())[0]
            target_path = f"{sub_group_path}/{first_frame_key}"
            print(f"--> Fallback auto-detected camera dataset path: '{target_path}'")
            return target_path

    return None


def extract_and_split_lpbf_data(hdf5_file_paths, output_base, val_split=0.15, test_split=0.15, binary=False):
    """
    Parses L-PBF HDF5 files into U-Net datasets by auto-detecting internal camera channels.
    """
    setup_directories(output_base)
    all_layers_data = []

    # 1. Collect references to all existing layers across files
    for file_path in hdf5_file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        with h5py.File(file_path, 'r') as f:
            # Find the actual camera path inside this specific file
            desired_cams = ['visible', 'visible light']
            camera_dataset_path = find_camera_dataset_path(f, desired_cams)

            if not camera_dataset_path:
                print(f"Error: Could not find valid camera dataset inside 'slices/camera_data' for {file_path}.")
                if 'slices/camera_data' in f:
                    print(f"Available sub-keys: {list(f['slices/camera_data'].keys())}")
                continue

            build_name = f.attrs.get('core/build_name', os.path.basename(file_path).replace('.hdf5', ''))

            # Read shape: [total_layers, height, width]
            total_layers = f[camera_dataset_path].shape[0]
            print(f"Found build '{build_name}' with {total_layers} layers available.")

            # Extract class names safely from the metadata
            class_names = []
            if 'slices/segmentation_results' in f:
                class_names = f['slices/segmentation_results'].attrs['class_names'].split(",")
                # class_names = [c.decode('utf-8') if isinstance(c, bytes) else c
                #                for c in f['slices/segmentation_results/class_names'][:]]

            for layer_idx in range(total_layers):
                all_layers_data.append({
                    'file_path': file_path,
                    'camera_path': camera_dataset_path,
                    'layer_idx': layer_idx,
                    'build_name': build_name,
                    'class_names': class_names
                })

    if not all_layers_data:
        print("No layers were extracted. Dataset construction aborted.")
        return

    # 2. Shuffle and split deterministically
    np.random.seed(42)
    np.random.shuffle(all_layers_data)

    total_samples = len(all_layers_data)
    num_test = int(total_samples * test_split)
    num_val = int(total_samples * val_split)

    test_set = all_layers_data[:num_test]
    val_set = all_layers_data[num_test:num_test + num_val]
    train_set = all_layers_data[num_test + num_val:]

    splits_mapping = {'train': train_set, 'val': val_set, 'test': test_set}

    # 3. Process and export data arrays safely
    for split_name, dataset in splits_mapping.items():
        print(f"Processing {len(dataset)} layers for the '{split_name}' split...")

        for item in dataset:
            with h5py.File(item['file_path'], 'r') as f:
                idx = item['layer_idx']
                cam_path = item['camera_path']

                # Slice the 3D array at the layer dimension index
                img_array = f[cam_path][idx, ...]

                # Initialize an empty mask array matching image dimensions
                mask_array = np.zeros(img_array.shape, dtype=np.uint8)

                # # Process anomaly maps
                # # Class 0 remains Background / Normal Powder Bed
                # for class_id in range(len(item['class_names'])):
                #     class_key = f"slices/segmentation_results/{class_id}"
                #     if class_key in f:
                #         binary_mask = f[class_key][idx, ...]
                #         mask_array[binary_mask == 1] = class_id + 1

                # for file_class_id, class_name in enumerate(item['file_class_names']):
                for file_class_id, class_name in enumerate(item['class_names']):
                    class_key = f"slices/segmentation_results/{file_class_id}"

                    if class_key in f:
                        # Extract the target binary matrix slice
                        binary_mask = f[class_key][idx, ...]

                        # Fetch the universal standardized integer ID
                        standard_id = standardized_class_dict_2025.get(class_name, -1)
                        if standard_id == -1:
                            if np.any(binary_mask == 1): # There is an undocumented class present in the data that does not exist in the standardized_class_dict
                                print(f"class_name {class_name} inconsistent with standardized_class_dict.")
                                print(f"Please update the standardized_class_dict to include the class.")
                                sys.exit(1)
                            else: # Class is not represented in the data -- continue
                                continue

                        if standard_id != 0:
                            if binary:
                                mask_array[binary_mask == 1] = 1
                            else:
                                mask_array[binary_mask == 1] = standard_id

                # Sanity-check to ensure binary mask
                if binary:
                    assert(list(np.unique(mask_array)) == [0,1] or list(np.unique(mask_array)) == [0])

                # Generate a clean unique filename
                file_filename = f"{item['build_name']}_layer_{idx:05d}"

                # Save Image as high-fidelity .tif
                img_out_path = os.path.join(output_base, split_name, 'images', f"{file_filename}.tif")
                tiff.imwrite(img_out_path, img_array)

                # Save Mask as lightweight .png
                mask_out_path = os.path.join(output_base, split_name, 'masks', f"{file_filename}.png")
                Image.fromarray(mask_array).save(mask_out_path)


if __name__ == "__main__":
    # Load configuration parameters
    binary = config.BINARY_CLASSIFICATION

    # Point directly to your true local path directory shown in your terminal output
    lpbf_files = [
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-05-09 M2 Anomaly Detection Print 01.hdf5",
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-05-13 M2 Stripe Rotation Print 01.hdf5",
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-06-11 M2 Globe.hdf5"
    ]

    # file_breakdowns, total_dataset_distribution = count_distinct_anomaly_instances(lpbf_files)

    output_directory = "D:/Data/Peregrine/Unified_Unet_Dataset"
    extract_and_split_lpbf_data(lpbf_files, output_directory, binary=binary)
    print("Parsing completed successfully!")
