import numpy as np
import cv2
import matplotlib.pyplot as plt

#Loading the image to be tested
test_image = cv2.imread('data/baby.jpeg')
podium_people = cv2.imread('data/podium_people.png')
podium_people2 = cv2.imread('data/podium_people2.png')
# print(test_image)
# if (test_image == None):
#     print("Image not define")

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        
    return image_copy


faces = detect_faces(haar_cascade_face, test_image)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))
plt.show()