# Mask Segmentation - INRIA Project

## Dataset

You can download the dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

In '**Align&Cropped Images**' Drive download '**Img/img_align_celeba.zip**' and extract it in the a new folder called **data** at the root of the project.

## MaskTheFace

We are using [MasktheFace project](https://github.com/aqeelanwar/MaskTheFace) to add covid mask on celeba dataset to create labels from the created picture.

## 🚀&nbsp; Installation

- Refer to principal README, section installation.

## 🧑🏻‍💻&nbsp; Train

- If you want to restart training of the segmentation model, be sure to be in mask_segmentation directory:

```bash
git clone https://github.com/aqeelanwar/MaskTheFace.git
sed -i 's/ utils./ MaskTheFace.utils./' MaskTheFace/utils/aux_functions.py
python train.py
```
