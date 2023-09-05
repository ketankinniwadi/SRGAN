import cv2
import numpy as np
import os

for img in os.listdir("train"):
    img_array = cv2.imread('train/{}'.format(img))

    img_array = cv2.resize(img_array, (128, 128))
    lr_img_array = cv2.resize(img_array, (96, 96))

    cv2.imwrite("hr_image/"+img, img_array)
    cv2.imwrite("lr_image/"+img, lr_img_array)


