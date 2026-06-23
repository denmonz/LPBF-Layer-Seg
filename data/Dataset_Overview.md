# Dataset Overview
The data used for model training, validation, and testing are the Laser Powder Bed Fusion builds within the three datasets. 
I utilized the visible imagery ('0'), which are single-channel .tif images. The annotations generated are .png files, where 
each pixel represents the class associated with that pixel (for binary data, 0 = background / no defect, and 1 = defect, 
for multiclass data, 0 = background / no defect, and >0 is the actual class value [1-17]).

## Classes
| Class Id | Class Name               | Distinct Polygon Instances | % of Total Pixels |
|:---------|:-------------------------|----------------------------|-------------------|
| 0        | Unlabeled                | 372                        | 0.06%             |
| 0        | Powder                   | 23,353                     | 82.89%            |
| 0        | Printed                  | 17,148                     | 7.39%             |
| 0        | Misprint                 | 2,189                      | 0.01%             |
| 1        | Recoaster Hopping        | 11,652                     | 0.05%             |
| 2        | Recoater Streaking       | 849                        | 0.26%             |
| 3        | Incomplete Spreading     | 155                        | 0.40%             |
| 4        | Edge Swelling            | 95,449                     | 0.04%             |
| 5        | Debris                   | 4,782                      | 0.01%             |
| 6        | Super Elevation          | 4,482                      | 0.09%             |
| 7        | Soot                     | 37,768                     | 0.07%             |
| 8        | Excessive / Over Melting | 11,097                     | 0.33%             |
| 9        | Spatter                  | 48,807                     | 0.02%             |
| 10       | Localized Bright Spot    | 4,275                      | 0.00%             |
| 11       | Localized Dark Region    | 0                          | -                 |
| 12       | Stripe Boundary          | 9,682                      | 0.11%             |
| 13       | Under Melting            | 1,488                      | 0.11%             |
| 14       | Recoater Flicking        | 15,795                     | 0.05%             |
| 15       | Disturbed Powder         | 4,971                      | 0.24%             |
| 16       | Recoater Strike Damage   | 2,367                      | 0.00%             |
| 17       | Condensate               | 12,960                     | 7.89%             |
| 18       | Dropped NIR Data         | 0                          | -                 |

### Class Distribution Notes
- The **most** represented classes (by pixel percentage) are:
   1. Powder (82.89%)
   2. Condensate (7.89%)
   3. Printed (7.39%)
   4. Incomplete Spreading (0.40%)
   5. Excessive / Over Melting (0.33%)


- The **least** represented classes (by pixel percentage) are:
   1. Spatter (0.02%)
   2. Misprint (0.01%)
   3. Debris (0.01%)
   4. Localized Bright Spot (0.00%)
   5. Recoater Strike Damage (0.00%)


- 'Localized Dark Region' and 'Dropped NIR Data' are not contained within the subset of the data


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