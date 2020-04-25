import os
import cv2
from PIL import Image
import numpy as np

data=[]
labels=[]

reds=os.listdir("images/reds")
for red in reds:
    imag=cv2.imread("images/reds/"+red)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

browns=os.listdir("images/browns")
for brown in browns:
    imag=cv2.imread("images/browns/"+brown)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)


# Convert to NP Arrays
animals = np.array(data)
labels = np.array(labels)
np.save("labels/animals",animals)
np.save("labels/labels",labels)

def main():
    print("Hello World")