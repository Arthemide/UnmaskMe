# dataset creation

## celebA

We are using celebA dataset 202 000 image

## MaskTheFace

we are using MasktheFace project:
<https://github.com/aqeelanwar/MaskTheFace>
to add covid mask on celeba dataset to create labels from the created picture.

## labels creation

to create the labels we must compare the masked picture but because MasktheFace adapt the luminosity we cant juste compare pixel by pixel

## inria unmaskme

runned on kaggle a unet models to segmentate the mask

## unet_tensorflow/pytorch

useless for now
