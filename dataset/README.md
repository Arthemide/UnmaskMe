# Dataset

This use a rewritten version of the [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace) to augment the data on the
fly

## Usage

```python
from dataset.data_augmentation import MaskedFaceDataset

dataset = MaskedFaceDataset(
    root_dir='/path/to/dataset',
)
```

Optional arguments:

- ``mask_type``: list of strings, accepted and default value : ["surgical", "N95", "KN95", "cloth", "gas", "inpaint"].
  The type of mask to use.


- ``pre_transform``: callable, default None. A function/transform that takes in an PIL image and returns a transformed
  version, to be used before the mask is applied. E.g, ``transforms.RandomCrop``


- ``post_transform``: callable, default None. A function/transform that takes in an PIL image and returns a transformed
  version, to be used after the mask is applied. E.g, ``transforms.RandomCrop``

