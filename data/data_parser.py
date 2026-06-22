import cv2
import matplotlib.pyplot as plt
import numpy as np
import pprint
import shutil
from pathlib import Path
from scipy.ndimage import label
from data.peregrine2021_metadata import class_dict_2021
from data.peregrine2022_metadata import class_dict_2022, standardized_class_dict_2022, non_represented_classes
from configs import config


def visualize_standardized_mask(mask: np.ndarray):
    color_map = {
        -1: [0.2, 0.2, 0.2], 0: [0.9, 0.9, 0.9], 1: [0.7, 0.9, 0.7],
        2: [1.0, 0.0, 0.0], 3: [1.0, 0.6, 0.0], 4: [1.0, 1.0, 0.0],
        5: [0.0, 0.0, 1.0], 6: [0.5, 0.0, 0.5], 7: [0.0, 0.8, 0.8],
        8: [0.6, 0.4, 0.2], 9: [0.9, 0.1, 0.5], 10: [1.0, 0.4, 0.4],
        11: [0.1, 0.6, 0.2], 12: [0.0, 0.4, 0.8], 13: [0.8, 0.8, 0.0],
        14: [0.3, 0.3, 0.5], 15: [0.5, 0.5, 0.0], 16: [0.6, 0.2, 0.8],
    }
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3))
    unique_classes_present = np.unique(mask)
    for class_id in unique_classes_present:
        if class_id in color_map:
            rgb_image[mask == class_id] = color_map[class_id]
        else:
            rgb_image[mask == class_id] = [0.0, 0.0, 0.0]
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.axis("off")
    legend_elements = []
    for class_id in unique_classes_present:
        if class_id in standardized_class_dict_2022:
            name = standardized_class_dict_2022[class_id]
            color = color_map.get(class_id, [0.0, 0.0, 0.0])
            patch = plt.Line2D([0], [0], marker="s", color="none", label=name, markerfacecolor=color, markersize=10)
            legend_elements.append(patch)
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), title="Annotation Classes")
    plt.tight_layout()
    plt.show()


def count_class_instances(mask: np.ndarray) -> dict:
    instance_counts = {}
    for class_name, class_id in standardized_class_dict_2022.items():
        class_mask = mask == class_id
        _, num_features = label(class_mask, structure=np.ones((3, 3)))
        if num_features > 0:
            instance_counts[class_name] = num_features
    return instance_counts


def standardize_mask(mask: np.ndarray, printer: str, material: str) -> np.ndarray:
    try:
        original_mapping = (class_dict_2021 | class_dict_2022)[printer][material]
    except KeyError as e:
        raise KeyError(
            f"Missing configuration entry for Printer: '{printer}', Material: '{material}' in class_dict.") from e
    standard_lookup = {label: std_id for std_id, label in standardized_class_dict_2022.items()}
    translation_dict = {}
    for orig_id, label_name in original_mapping.items():
        if label_name in non_represented_classes:
            continue
        elif label_name in standardized_class_dict_2022.keys():
            translation_dict[orig_id] = standardized_class_dict_2022[label_name]
        else:
            raise ValueError(
                f"Label '{label_name}' is missing from both standardized_class_dict_2022 and non_represented_classes.")
    standardized_mask = np.full_like(mask, fill_value=-1, dtype=np.int32)
    for orig_id, std_id in translation_dict.items():
        standardized_mask[mask == orig_id] = std_id
    return standardized_mask


# --- MODIFIED CONVERSION FUNCTION FOR U-NET RASTER MASKS ---
def mask_to_unet_segmentation(mask_path, output_png_path, binary=False, visualize=False):
    """Converts a single .npy mask into a rasterized Grayscale .png image for U-Net."""
    mask = np.load(mask_path)

    if "2022" in str(Path(mask_path)):
        # Retrieve the printer and material names from path context
        printer_name = str(Path(mask_path).parents[3].name)
        material_name = str(Path(mask_path).parents[2].name)
    else:
        printer_name = "ConceptLaser_M2"
        material_name = "316L_Stainless_Steel"

    # Standardize classes
    standardized_mask = standardize_mask(mask, printer_name, material_name)

    if visualize:
        visualize_standardized_mask(standardized_mask)

    standardized_instance_counts = count_class_instances(standardized_mask)
    binary_instance_counts = None

    if binary:
        """
        Combine "unlabeled" (-1), "powder" (0), "printed" (1) into class 0.
        Combine all other structural defect variations into class 1.
        """
        mapped_mask = np.full_like(standardized_mask, 0, dtype=np.uint8)  # Default everything to background (0)
        # Identify any defect pixel value (everything >= 1) and label it as 1
        mapped_mask[(standardized_mask >= 1)] = 1

        # Calculate instances on the binary mapped matrix
        # (Note: We use a temporary dictionary mapping for count_class_instances compatibility)
        _, defect_count = label(mapped_mask == 1, structure=np.ones((3, 3)))
        _, no_defect_count = label(mapped_mask == 0, structure=np.ones((3, 3)))
        binary_instance_counts = {"no_defect": no_defect_count, "defect": defect_count}

        if visualize:
            visualize_standardized_mask(mapped_mask)
    else:
        # For Multi-class U-Net, shift unlabeled (-1) to a safe unsigned integer (0 or 255)
        # based on loss configuration. Here we preserve absolute Class IDs.
        mapped_mask = np.where(standardized_mask == -1, 0, standardized_mask).astype(np.uint8)

    # Save the 2D matrix directly as a 1-channel grayscale image
    cv2.imwrite(str(output_png_path), mapped_mask)

    return standardized_instance_counts, binary_instance_counts

def build_partitioned_unet_dataset(dataset_configs, unet_output_dir, binary=True, visualize=False):
    """
    Parses multiple versions of Peregrine dataset hierarchies, separating them
    into isolated train/val/test folders to guarantee zero cross-material data leakage.

    dataset_configs: List of dicts specifying root directory paths and version flags.
    """
    unet_root = Path(unet_output_dir)

    # Strictly reserve these specific material groups for the test bench to evaluate cross-material generalization
    test_groups = ["Maraging_Steel", "Inconel_718_2"]

    # Create target folder structural tree
    for split in ["train", "val"]:
        (unet_root / split / "images").mkdir(parents=True, exist_ok=True)
        (unet_root / split / "masks").mkdir(parents=True, exist_ok=True)

    total_original_instances_count_dict = {
        # "unlabeled": 0,
        "powder": 0,
        "printed": 0,
        "recoater_hopping": 0,
        "recoater_streaking": 0,
        "incomplete_spreading": 0,
        "edge_swelling": 0,
        "debris": 0,
        "super_elevation": 0,
        "soot": 0,
        # "misprint": 0,
        "excessive_melting": 0,
        # "crashing": 0,
        "spatter": 0,
        "localized_bright_spot": 0,
        "localized_dark_regions": 0,
        # "mounding": 0,
        "stripe_boundary": 0
    }

    total_binary_instances_count_dict = {
        "no_defect": 0,
        "defect": 0
    }

    for config in dataset_configs:
        source_root = Path(config["root_dir"])
        version = config["version"]

        print(f"\n==========================================")
        print(f"PROCESSING MASTER DATASET: {version} via {source_root}")
        print(f"==========================================")

        if not source_root.exists():
            print(f"Warning: Source path {source_root} does not exist. Skipping...")
            continue

        # Find all annotations folders to anchor our crawl
        annotation_folders = list(source_root.glob("**/annotations"))
        print(f"Found {len(annotation_folders)} annotations subfolders in this root.")

        for ann_folder in annotation_folders:
            # 1. Resolve naming context and hierarchy variants based on version
            if version == "2022":
                # Path template: source_root / [printer] / [material] / training / annotations
                printer_name = ann_folder.parent.parent.parent.name
                material_name = ann_folder.parent.parent.name
                img_folder = ann_folder.parent / "data" / "visible" / "0"
            elif version == "2021":
                # Only one printer and material present in dataset
                printer_name = "ConceptLaser_M2"
                material_name = "316L_Stainless_Steel"
                img_folder = ann_folder.parent / "data" / "visible" / "0"
            else:
                print(f"Unknown dataset configuration version: {version}")
                continue

            prefix = f"v{version}_{printer_name}_{material_name}"

            if not img_folder.exists():
                print(f"Warning: Corresponding image directory missing for {prefix}, skipping...")
                continue

            npy_files = sorted(list(ann_folder.glob("*.npy")))
            total_files = len(npy_files)
            if total_files == 0:
                continue

            # 2. Determine Split Bucket assignment
            if material_name in test_groups:
                print(f"Routing {prefix} ({total_files} slices) -> HELD OUT FOR TEST BENCH")
                # Create a dedicated path structure: test / printer_material
                test_subdir = Path("test") / f"{printer_name}_{material_name}"
                file_assignments = [(f, test_subdir) for f in npy_files]
            else:
                # Dynamically split remaining materials 80% train / 20% val by layer slice index
                split_idx = int(0.8 * total_files)
                print(f"Routing {prefix} -> Distributed into: {split_idx} Train | {total_files - split_idx} Val")
                file_assignments = [(f, "train") for f in npy_files[:split_idx]] + \
                                   [(f, "val") for f in npy_files[split_idx:]]

            # 3. Convert masks and distribute files
            for npy_path, split_name in file_assignments:
                file_id = npy_path.stem
                tif_path = img_folder / f"{file_id}.tif"
                if not tif_path.exists():
                    continue

                unique_name = f"{prefix}_{file_id}"

                dest_mask = unet_root / split_name / "masks" / f"{unique_name}.png"
                dest_image = unet_root / split_name / "images" / f"{unique_name}.tif"

                dest_mask.parent.mkdir(parents=True, exist_ok=True)
                dest_image.parent.mkdir(parents=True, exist_ok=True)

                # Execute metadata standardization logic and matrix write transformations
                standardized_instance_counts, binary_instance_counts = mask_to_unet_segmentation(npy_path, dest_mask, binary, visualize)
                shutil.copy2(tif_path, dest_image)

                for k, v in standardized_instance_counts.items():
                    total_original_instances_count_dict[k] += v

                # print("\nInstances Dictionary count:")
                # pprint.pprint(original_instance_counts, indent=4)

                if binary:
                    for k, v in binary_instance_counts.items():
                        total_binary_instances_count_dict[k] += v

    print("\nTotal multi-class instances dictionary count:")
    pprint.pprint(total_original_instances_count_dict, indent=4)

    if binary:
        print("\nTotal binary instances dictionary count:")
        pprint.pprint(total_binary_instances_count_dict, indent=4)

    print(f"\nUnified partitioned U-Net folder tree fully built at: {unet_root.resolve()}")

def main():
    visualize = False

    # Read the configuration file
    binary = config.BINARY_CLASSIFICATION  # Keep true for binary defect isolation
    dataset_base_dir = config.DATASET_BASE_DIR

    # Register all source pools with their structural classification tags
    dataset_configs = [
        {
            "root_dir": f"{dataset_base_dir}/Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/",
            "version": "2022"
        },
        {
            "root_dir": f"{dataset_base_dir}/Peregrine Dataset v2021-03/Laser Powder Bed Fusion/",
            "version": "2021"
        }
    ]

    # Destination directory where train/val/test folders will be generated
    unet_output_dir = config.UNET_DATASET_DIR

    build_partitioned_unet_dataset(dataset_configs, unet_output_dir, binary, visualize)


if __name__ == "__main__":
    main()