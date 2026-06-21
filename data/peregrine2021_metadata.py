class_dict_2021 = {
    "ConceptLaser_M2": {
        "316L_Stainless_Steel": {
            -1: "unlabeled",
            0: "powder",
            1: "printed",
            2: "recoater_hopping",
            3: "recoater_streaking",
            4: "incomplete_spreading",
            5: "edge_swelling",
            6: "debris",
            7: "super_elevation",
            8: "soot",
        }
    }
}

standardized_class_dict_2021 = {
    "unlabeled": 0,
    "powder": 0,
    "printed": 0,
    "recoater_hopping": 1,
    "recoater_streaking": 2,
    "incomplete_spreading": 3,
    "edge_swelling": 4,
    "debris": 5,
    "super_elevation": 6,
    "soot": 7,
}

binary_class_dict = {
    0: [0], # "No Defect"
    1: [1, 2, 3, 4, 5, 6, 7] # "Defect"
}
