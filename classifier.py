import os
import cv2
from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils

SAVE_LABELS = True
LOAD_LABELS = True

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

if SAVE_LABELS:
    np.save("labels/animals",animals)
    np.save("labels/labels",labels)

if LOAD_LABELS:
    animals=np.load("labels/animals.npy")
    labels=np.load("labels/labels.npy")    

s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

num_classes=len(np.unique(labels))
data_length=len(animals)

(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)