import configs.config as config
from data.data_parser import build_partitioned_unet_dataset
from data.data_parser_v2025_09 import extract_and_split_lpbf_data

def main():
    """
    Standardize the three datasets:
        Peregrine Dataset v2021-03 /
        Peregrine Dataset v2022-10.1 /
        Peregrine Dataset v2025-09 /
    into an ultimate train/val/test dataset:
        Unified_Unet_Dataset/
    """
    print("Standardizing the datasets...")
    BINARY_CLASSIFICATION = config.BINARY_CLASSIFICATION
    DATASET_BASE_DIR = config.DATASET_BASE_DIR
    UNET_DATASET_DIR = config.UNET_DATASET_DIR

    # First, start with the v2021-03 and v2022-10.1 datasets
    print("\tStandardizing the v2021-03 and v2022-10.1 datasets...")
    dataset_configs = [
        {
            "root_dir": f"{DATASET_BASE_DIR}/Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/",
            "version": "2022"
        },
        {
            "root_dir": f"{DATASET_BASE_DIR}/Peregrine Dataset v2021-03/Laser Powder Bed Fusion/",
            "version": "2021"
        }
    ]
    build_partitioned_unet_dataset(dataset_configs, UNET_DATASET_DIR, binary=BINARY_CLASSIFICATION)

    # Next, the v2025-09 dataset, which is different from the previous two and requires its own separate setup script
    print("\tStandardizing the v2025-09 dataset...")
    lpbf_files = [
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-05-09 M2 Anomaly Detection Print 01.hdf5",
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-05-13 M2 Stripe Rotation Print 01.hdf5",
        "D:/Data/Peregrine/Peregrine Dataset v2025-09/Concept Laser M2 Builds/2025-06-11 M2 Globe.hdf5"
    ]
    extract_and_split_lpbf_data(lpbf_files, UNET_DATASET_DIR, binary=BINARY_CLASSIFICATION)
    return

if __name__ == "__main__":
    main()