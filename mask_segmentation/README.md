# Mask Segmentation

## Dataset

You can download the dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

In '**Align&Cropped Images**' Drive download '**Img/img_align_celeba.zip**' and extract it in the a new folder called **data** at the root of the project.

## MaskTheFace

we are using MasktheFace [project](https://github.com/aqeelanwar/MaskTheFace):
to add covid mask on celeba dataset to create labels from the created picture.

## Model / utils

model.py and utils.py are for the main function.

## Train

if you want to retrain the segmentation model you can simply run:

``` bash
python train.py
```
