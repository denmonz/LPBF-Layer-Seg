# Dataset Overview
The data used for model training, validation, and testing are the Laser Powder Bed Fusion builds within the three datasets. 
I utilized the visible imagery ('0'), which are single-channel .tif images. The annotations generated are .png files, where 
each pixel represents the class associated with that pixel (for binary data, 0 = background / no defect, and 1 = defect, 
for multiclass data, 0 = background / no defect, and >0 is the actual class value [1-17]).

## Classes
| Class Id | Class Name               | Number of Instances |
|:---------|:-------------------------|---------------------|
| 0        | Unlabeled                | 372                 |
| 0        | Powder                   | 23,353              |
| 0        | Printed                  | 17,148              |
| 0        | Misprint                 | 2,189               |
| 1        | Recoaster Hopping        | 11,652              |
| 2        | Recoater Streaking       | 849                 |
| 3        | Incomplete Spreading     | 155                 |
| 4        | Edge Swelling            | 95,449              |
| 5        | Debris                   | 4,782               |
| 6        | Super Elevation          | 4,482               |
| 7        | Soot                     | 37,768              |
| 8        | Excessive / Over Melting | 11,097              |
| 9        | Spatter                  | 48,807              |
| 10       | Localized Bright Spot    | 4,275               |
| 11       | Localized Dark Region    | 0                   |
| 12       | Stripe Boundary          | 9,682               |
| 13       | Under Melting            | 1,488               |
| 14       | Recoater Flicking        | 15,795              |
| 15       | Disturbed Powder         | 4,971               |
| 16       | Recoater Strike Damage   | 2,367               |
| 17       | Condensate               | 12,960              |

### Class Distribution Notes
- The **most** represented classes are:
   1. Edge Swelling (95,4949)
   2. Spatter (48,807)
   3. Soot (37,768)


- The **least** represented classes are (not including 'Unlabeled'):
   1. Incomplete Spreading (155)
   2. Recoater Streaking (849)
   3. Under Melting (1,488)


- 'Localized Dark Region' is not contained within the subset of the data


## Peregrine Dataset v2021-03
### Builds
**Train (80%) / Val (20%) Data:**

`Laser Powder Bed Fusion`


## Peregrine Dataset v2022-10.1
### Builds
**Train (80%) / Val (20%) Data:**

`EOS_M290 > 17-4_PH_Stainless_Steel`

`EOS_M290 > GammaPrint-700`

`EOS_M290 > Inconel_718_1`

**Test (100%) Data:**

`AddUp_FormUp_350 > Maraging_Steel`

`EOS_M290 > Inconel_718_2`


## Peregrine Dataset v2025-09
### Builds
**Train (70%) / Val (15%) / Test (15%) Data:**

`Concept Laser M2 Builds > 2025-05-09 M2 Anomaly Detection Print 01.hdf5`

`Concept Laser M2 Builds > 2025-05-13 M2 Stripe Rotation Print 01.hdf5`

`Concept Laser M2 Builds > 2025-06-11 M2 Globe.hdf5`

## General Notes
- There are **many** differences between Class IDs between each dataset, and within the datasets builds themselves, which 
required standardization across all builds.
- Some classes are not represented in the data at all, and are noted in the individual `*_metadata.py` files.
- The `Peregrine Dataset v2021-03` and `Peregrine Dataset v2022-10.1` datasets have a small number of images that contained 
a **dense** amount of annotations per image.
- The `Peregrine Dataset v2025-09` has an entirely different structure than the previous two datasets, and required its 
own specific means of data extraction.