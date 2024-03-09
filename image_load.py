from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import models

from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
img=image.load_img("C:\\Users\\udayk\\Downloads\\ASL_Dataset\\Train\\A\\A1976.jpg")
plt.imshow(img)
plt.show()
print(cv2.imread("C:\\Users\\udayk\\Downloads\\ASL_Dataset\\Train\\A\\A1976.jpg"))
train=ImageDataGenerator(rescale=1/163)    # (400, 400, 3)   normalize by 163
validation=ImageDataGenerator(rescale=1/163)
train_dataset=train.flow_from_directory("C:\\Users\\udayk\\OneDrive\\AppData\\Desktop\\signlanguage_dataset\\training",target_size=(400,400),batch_size=32,class_mode='categorical')
validation_dataset=validation.flow_from_directory("C:\\Users\\udayk\\OneDrive\\AppData\\Desktop\\signlanguage_dataset\\validation",target_size=(400,400),batch_size=32,class_mode='categorical')
# #           print(train_dataset.class_indices) #Found 165670 images belonging to 28 classes.
# #            print(validation_dataset.class_indices) #Found 165670 images belonging to 28 classes.
