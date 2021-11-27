
import cv2
import numpy

def replace_face(image, gan_preds, locations):
  for (box, pred) in zip(locations, gan_preds):
    (startX, startY, endX, endY) = box
    image[startY:endY, startX:endX]= pred
  return image