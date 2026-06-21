class_dict_2025 = {
    0: "powder",
    1: "printed",
    2: "printed_mesh",
    3: "misprint",
    4: "excessive_melting",
    5: "under_melting",
    6: "stripe_boundary",
    7: "recoater_flicking",
    8: "recoater_streaking",
    9: "incomplete_spreading",
    10: "disturbed_powder",
    11: "recoater_strike_damage",
    12: "super_elevation",
    13: "edge_swelling",
    14: "pitting", # Non-represented class - IGNORE
    15: "spatter",
    16: "condensate",
    17: "localized_bright_spot",
    18: "localized_dark_spot", # Non-represented class - IGNORE
    19: "dropped_NIR_data" # Non-represented class - IGNORE
}

"""
ORIGINAL DATA NOTES

Class Rename
    - "over melting" -> "excessive_melting"
    - "super-elevation" -> "super_elevation"
    - "swelling" -> "edge_swelling"
"""

standardized_class_dict_2025 = {
    "Powder": 0,
    "Printed": 0,
    "Printed Mesh": 0,
    "Misprint": 0,
    "Over Melting": 8,
    "Under Melting": 13,
    "Stripe Boundary": 13,
    "Recoater Flicking": 14,
    "Recoater Streaking": 3,
    "Incomplete Spreading": 4,
    "Disturbed Powder": 15,
    "Recoater Strike Damage": 16,
    "Super-Elevation": 7,
    "Swelling": 5,
    "Spatter on Powder": 10,
    "Condensate": 17,
    "Localized Bright Spot": 11
}

non_represented_classes = [
    "Pitting",
    "Localized Dark Spot"
    "Dropped NIR Data"
]

binary_class_dict = {
    0: [0], # "No Defect"
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # "Defect"
}

"""
NOTES
    
Class Merge
    - (1) - "printed", "printed mesh", and "misprint"

The following are empty classes, and are not represented in the training data:
    - "pitting"
    - "localized_dark_spot"
    - "dropped_NIR_data"
"""