class_dict_2022 = {
    "AddUp_FormUp_350": {
        "Maraging_Steel": {
            -1: "unlabeled", # Non-represented class - IGNORE
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "edge_swelling",
            6: "debris",
            7: "super_elevation",
            8: "soot",
            9: "misprint" # Non-represented class - IGNORE
        }
    },
    "EOS_M290" : {
        "17-4_PH_Stainless_Steel": {
            -1: "unlabeled", # Non-represented class - IGNORE
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "edge_swelling",
            6: "debris",
            7: "super_elevation",
            8: "soot",
            9: "excessive_melting",
            10: "crashing", # Non-represented class - IGNORE
            11: "misprint" # Non-represented class - IGNORE
        },
        "GammaPrint-700": {
            -1: "unlabeled", # Non-represented class - IGNORE
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "debris",
            6: "edge_swelling",
            7: "super_elevation",
            8: "spatter",
            9: "localized_bright_spot",
            10: "mounding", # Non-represented class - IGNORE
            11: "stripe_boundary",
            12: "excessive_melting",
            13: "misprint", # Non-represented class - IGNORE
            14: "localized_dark_regions"
        },
        "Inconel_718_1": {
            -1: "unlabeled", # Non-represented class - IGNORE
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "debris",
            6: "edge_swelling",
            7: "super_elevation",
            8: "spatter",
            9: "localized_bright_spot",
            10: "mounding", # Non-represented class - IGNORE
            11: "stripe_boundary",
            12: "excessive_melting",
            13: "misprint", # Non-represented class - IGNORE
            14: "localized_dark_regions"
        },
        "Inconel_718_2": {
            -1: "unlabeled", # Non-represented class - IGNORE
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "edge_swelling",
            6: "debris",
            7: "super_elevation",
            8: "soot",
            9: "excessive_melting",
            10: "crashing", # Non-represented class - IGNORE
            11: "misprint" # Non-represented class - IGNORE
        }
    }
}

standardized_class_dict_2022 = {
    "powder": 0,
    "printed": 0,
    "recoater_hopping": 1,
    "recoater_streaking": 2,
    "incomplete_spreading": 3,
    "edge_swelling": 4,
    "debris": 5,
    "super_elevation": 6,
    "soot": 7,
    "excessive_melting": 8,
    "spatter": 9,
    "localized_bright_spot": 10,
    "localized_dark_regions": 11,
    "stripe_boundary": 12
}

non_represented_classes = [
    "unlabeled",
    "crashing",
    "mounding",
    "misprint"
]

binary_class_dict = {
    0: [0], # "No Defect"
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # "Defect"
}

"""
The following are empty classes, and are not represented in the training data:
    - "crashing"
    - "misprint"
    - "mounding"
    - "unlabeled"
"""