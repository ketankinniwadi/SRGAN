import cv2
import numpy as np
import os

for img in os.listdir("source"):
    img_array = cv2.imread('source/{}'.format(img))

    img_array = cv2.resize(img_array, (384, 384))
    lr_img_array = cv2.resize(img_array, (96, 96))

    cv2.imwrite("hr_test/"+img, img_array)
    #cv2.imwrite("lr_test/"+img, lr_img_array)


