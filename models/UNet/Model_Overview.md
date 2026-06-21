# Model Overview
## Architecture Details

**Architecture:** U-Net++ from [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/models.html)

**Encoder:** EfficientNet ('efficientnet-b3') pretrained on the [ImageNet Dataset](https://www.image-net.org/)

**Classification Type:** Binary ['Defect' vs. 'No Defect']

## Loss Functions
A combination of Focal Loss (Weight: 0.6), and Log-Loss Dice (Weight: 0.4) were used to optimize for microscopic defects within prints.

### Focal Loss Component [[link](https://smp.readthedocs.io/en/latest/losses.html#focalloss)]
Provides dynamic background down-weighting and precision control.

### Log-Loss Dice Component [[link](https://smp.readthedocs.io/en/latest/losses.html#diceloss)]
Mitigates small-scale pixel penalties and log-scale smoothing.